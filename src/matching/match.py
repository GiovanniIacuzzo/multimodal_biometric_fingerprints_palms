import os
import numpy as np
import math
from typing import List, Tuple, Dict
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.matching.utils import rotate_points, angle_diff
import logging

logger = logging.getLogger(__name__)


def apply_rigid_transform(points_xy: np.ndarray, theta: float, t: np.ndarray) -> np.ndarray:
    """Applica rotazione theta e traslazione t a punti Nx2"""
    return rotate_points(points_xy, theta) + t


def compute_descriptor_weight(minutia: np.ndarray) -> float:
    # sicurezza: se mancano campi, fallback neutro
    q = float(minutia[4]) if len(minutia) > 4 else 0.0
    coh = float(minutia[5]) if len(minutia) > 5 else 0.0
    angs = float(minutia[6]) if len(minutia) > 6 else 0.0
    return float(np.clip(0.2 + 0.4 * q + 0.2 * coh + 0.2 * angs, 0.01, 1.0))


def estimate_transform_rigid_by_pair(p_a: np.ndarray, p_b: np.ndarray,
                                     orient_a: float, orient_b: float) -> Tuple[float, np.ndarray]:
    theta = angle_diff(orient_b, orient_a)
    rotated = rotate_points(p_a.reshape(1, 2), theta).reshape(2)
    t = p_b - rotated
    return theta, t


def match_with_transform(mins_a: np.ndarray, mins_b: np.ndarray,
                         theta: float, t: np.ndarray,
                         dist_thresh: float, orient_thresh: float,
                         weights_a: np.ndarray, weights_b: np.ndarray,
                         use_type: bool = True) -> Tuple[List[Tuple[int, int, float]], int]:
    if mins_a.shape[0] == 0 or mins_b.shape[0] == 0:
        return [], 0

    pts_a, pts_b = mins_a[:, :2], mins_b[:, :2]
    orients_a, orients_b = mins_a[:, 3], mins_b[:, 3]
    types_b = mins_b[:, 2] if use_type else np.zeros(mins_b.shape[0])

    transformed_a = apply_rigid_transform(pts_a, theta, t)
    tree_b = KDTree(pts_b)
    dists, idxs = tree_b.query(transformed_a, k=1)
    dists, idxs = dists.ravel(), idxs.ravel()

    inliers = []
    for ia, (d, ib) in enumerate(zip(dists, idxs)):
        if d > dist_thresh:
            continue
        if use_type and int(mins_a[ia, 2]) != int(types_b[ib]):
            continue
        ang_diff_val = abs(angle_diff(orients_a[ia] + theta, orients_b[ib]))
        if ang_diff_val > orient_thresh:
            continue
        # peso spaziale + orientamento + descrittore
        gauss = math.exp(-(d ** 2) / (2 * (dist_thresh / 2.0) ** 2))
        orient_factor = math.exp(-(ang_diff_val ** 2) / (2 * (orient_thresh / 2.0) ** 2))
        combined = float(gauss * weights_a[ia] * weights_b[ib] * orient_factor)
        inliers.append((ia, int(ib), combined))

    return inliers, len(inliers)


def ransac_worker(args):
    """
    args = (mins_a, mins_b, dist_thresh, orient_thresh, min_inliers,
            use_type, weights_a, weights_b, rng_seed)
    """
    mins_a, mins_b, dist_thresh, orient_thresh, min_inliers, use_type, weights_a, weights_b, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    nA, nB = mins_a.shape[0], mins_b.shape[0]
    idxs_a, idxs_b = np.arange(nA), np.arange(nB)

    best_local = {"theta": 0.0, "t": np.array([0.0, 0.0]), "inliers": [], "score": 0.0}

    # --- robust sampling con fallback se somme dei pesi == 0
    wa_sum = float(np.sum(weights_a))
    if wa_sum <= 0 or np.isnan(wa_sum):
        ia = int(rng.choice(idxs_a))
    else:
        probs_a = (weights_a / wa_sum)
        ia = int(rng.choice(idxs_a, p=probs_a))

    if use_type:
        candidates_b = idxs_b[mins_b[:, 2] == mins_a[ia, 2]]
        if candidates_b.size == 0:
            wb_sum = float(np.sum(weights_b))
            if wb_sum <= 0 or np.isnan(wb_sum):
                ib = int(rng.choice(idxs_b))
            else:
                ib = int(rng.choice(idxs_b, p=(weights_b / wb_sum)))
        else:
            pb = weights_b[candidates_b].astype(float)
            if pb.sum() <= 0 or np.isnan(pb.sum()):
                ib = int(rng.choice(candidates_b))
            else:
                pb = pb / pb.sum()
                ib = int(rng.choice(candidates_b, p=pb))
    else:
        wb_sum = float(np.sum(weights_b))
        if wb_sum <= 0 or np.isnan(wb_sum):
            ib = int(rng.choice(idxs_b))
        else:
            ib = int(rng.choice(idxs_b, p=(weights_b / wb_sum)))

    # Estimate transform
    theta, t = estimate_transform_rigid_by_pair(mins_a[ia, :2], mins_b[ib, :2],
                                                mins_a[ia, 3], mins_b[ib, 3])
    inliers, n_in = match_with_transform(mins_a, mins_b, theta, t,
                                         dist_thresh, orient_thresh,
                                         weights_a, weights_b, use_type)

    if n_in >= min_inliers:
        total_score = float(sum([c for (_, _, c) in inliers]))
        # NORMALIZZAZIONE basata sui pesi (robusta)
        denom = max(1e-8, min(weights_a.sum(), weights_b.sum()))
        normalized = float(total_score / denom)
        best_local = {"theta": theta, "t": t, "inliers": inliers, "score": normalized}

    return best_local


def ransac_align_and_match_parallel(mins_a, mins_b, dist_thresh,
                                   orient_thresh=math.radians(20.0),
                                   max_iter=300, min_inliers=6,
                                   use_type=True, stop_inlier_ratio=0.15,
                                   thread_workers: int = None):
    """
    Esegue RANSAC in parallelo (più tentativi) e restituisce il miglior risultato.
    - stop_inlier_ratio: percentuale di inliers rispetto al min(len(A), len(B)) per stop precoce.
    - thread_workers: limita il numero di thread locali.
    """
    if mins_a.shape[0] == 0 or mins_b.shape[0] == 0:
        return {"theta": 0.0, "t": np.array([0.0, 0.0]), "inliers": [], "score": 0.0, "inlier_ratio": 0.0}

    weights_a = np.array([compute_descriptor_weight(m) for m in mins_a], dtype=float)
    weights_b = np.array([compute_descriptor_weight(m) for m in mins_b], dtype=float)
    best = {"theta": 0.0, "t": np.array([0.0, 0.0]), "inliers": [], "score": 0.0}

    # Prepara args con semi deterministici
    base_seed = int(os.environ.get("MATCH_SEED", 42))
    args_list = [(mins_a, mins_b, dist_thresh, orient_thresh, min_inliers, use_type, weights_a, weights_b, base_seed + seed)
                 for seed in range(max_iter)]

    # Limita i thread per evitare oversubscription
    cpu_count = os.cpu_count() or 1
    max_threads = thread_workers if (thread_workers is not None) else min(8, max(1, cpu_count // 2))

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(ransac_worker, args) for args in args_list]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                logger.exception("Errore in ransac_worker: %s", e)
                continue

            if result and (result.get("score", 0.0) > best.get("score", 0.0)):
                best = result
                # early stop su base ratio inliers / min(len(A), len(B))
                if len(best["inliers"]) / max(1, min(len(mins_a), len(mins_b))) >= stop_inlier_ratio:
                    break

    # Refinement con inliers se sufficienti
    if len(best["inliers"]) >= 3:
        idxsA = np.array([i for i, _, _ in best["inliers"]], dtype=int)
        idxsB = np.array([j for _, j, _ in best["inliers"]], dtype=int)
        Pa, Pb = mins_a[idxsA, :2], mins_b[idxsB, :2]
        ca, cb = Pa.mean(axis=0), Pb.mean(axis=0)
        A_centered, B_centered = Pa - ca, Pb - cb
        H = A_centered.T.dot(B_centered)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T.dot(U.T)
        theta_ref = float(math.atan2(R[1, 0], R[0, 0]))
        t_ref = cb - rotate_points(ca.reshape(1, 2), theta_ref).reshape(2)
        inliers, n_in = match_with_transform(mins_a, mins_b, theta_ref, t_ref,
                                             dist_thresh, orient_thresh, weights_a, weights_b, use_type)
        total_score = float(sum([c for (_, _, c) in inliers]))
        denom = max(1e-8, min(weights_a.sum(), weights_b.sum()))
        normalized = float(total_score / denom)
        best = {"theta": theta_ref, "t": t_ref, "inliers": inliers, "score": normalized}

    best_n_in = len(best["inliers"])
    best["inlier_ratio"] = best_n_in / max(1, min(len(mins_a), len(mins_b)))
    best["score"] = float(np.clip(best["score"], 0.0, 1.0))
    return best


def match_minutiae_pair(mins_a, mins_b,
                        dist_thresh=25.0,
                        orient_thresh_deg=20.0,
                        use_type=True,
                        ransac_iter=300,
                        min_inliers=6,
                        stop_inlier_ratio=0.15,
                        cross_check=True,
                        thread_workers: int = 1,
                        debug: bool = False) -> Dict:
    """
    Restituisce un dict con:
      - final_score: similiarity normalizzata in [0,1]
      - inlier_ratio
      - matches (inliers)
      - theta, t
    - thread_workers: passato a ransac_align_and_match_parallel per limitare i thread.
    - debug: se True stampa info diagnostiche quando score == 0
    """
    if mins_a is None or mins_b is None:
        return {"final_score": 0.0, "inlier_ratio": 0.0, "matches": [], "theta": 0.0, "t": np.array([0.0, 0.0])}

    A, B = np.array(mins_a, copy=True), np.array(mins_b, copy=True)
    orient_thresh = math.radians(orient_thresh_deg)
    best = ransac_align_and_match_parallel(A, B, dist_thresh, orient_thresh,
                                           max_iter=ransac_iter, min_inliers=min_inliers,
                                           use_type=use_type, stop_inlier_ratio=stop_inlier_ratio,
                                           thread_workers=thread_workers)

    matches = best.get("inliers", [])

    if cross_check and matches:
        # ricontrollo simmetrico: trasformo A e controllo nearest-neighbor su transformed A
        transformed_A = apply_rigid_transform(A[:, :2], best["theta"], best["t"])
        treeA = KDTree(transformed_A)
        idx_b_to_a = treeA.query(B[:, :2], k=1)[1].ravel()
        # keep only mutual nearest neighbors
        matches = [(ia, ib, sc) for ia, ib, sc in matches if idx_b_to_a[ib] == ia]
        best["inliers"] = matches

    final_score = float(best.get("score", 0.0))
    inlier_ratio = float(best.get("inlier_ratio", 0.0))

    if debug and final_score <= 0.0:
        logger.info("DEBUG match failed: nA=%d nB=%d inliers=%d score=%g", A.shape[0], B.shape[0], len(matches), final_score)
        # mostra prime 3 minuzie per diagnosi rapida
        try:
            logger.debug("A[0:3]=%s", np.array2string(A[:3, :], precision=3, separator=","))
            logger.debug("B[0:3]=%s", np.array2string(B[:3, :], precision=3, separator=","))
        except Exception:
            pass

    return {"final_score": final_score,
            "inlier_ratio": inlier_ratio,
            "matches": matches,
            "theta": float(best.get("theta", 0.0)),
            "t": best.get("t", np.array([0.0, 0.0]))}
