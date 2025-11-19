import numpy as np
import math
from typing import List, Tuple, Dict
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.matching.utils import rotate_points, angle_diff


def apply_rigid_transform(points_xy: np.ndarray, theta: float, t: np.ndarray) -> np.ndarray:
    """Applica rotazione theta e traslazione t a punti Nx2"""
    return rotate_points(points_xy, theta) + t

def compute_descriptor_weight(minutia: np.ndarray) -> float:
    q, coh, angs = minutia[4], minutia[5], minutia[6]
    return float(np.clip(0.2 + 0.4*q + 0.2*coh + 0.2*angs, 0.01, 1.0))

def estimate_transform_rigid_by_pair(p_a: np.ndarray, p_b: np.ndarray,
                                     orient_a: float, orient_b: float) -> Tuple[float, np.ndarray]:
    theta = angle_diff(orient_b, orient_a)
    rotated = rotate_points(p_a.reshape(1,2), theta).reshape(2)
    t = p_b - rotated
    return theta, t

def match_with_transform(mins_a: np.ndarray, mins_b: np.ndarray,
                         theta: float, t: np.ndarray,
                         dist_thresh: float, orient_thresh: float,
                         weights_a: np.ndarray, weights_b: np.ndarray,
                         use_type: bool = True) -> Tuple[List[Tuple[int,int,float]], int]:
    if mins_a.shape[0] == 0 or mins_b.shape[0] == 0:
        return [], 0

    pts_a, pts_b = mins_a[:, :2], mins_b[:, :2]
    orients_a, orients_b = mins_a[:,3], mins_b[:,3]
    types_b = mins_b[:,2] if use_type else np.zeros(mins_b.shape[0])

    # Trasforma A
    transformed_a = apply_rigid_transform(pts_a, theta, t)
    tree_b = KDTree(pts_b)
    dists, idxs = tree_b.query(transformed_a, k=1)
    dists, idxs = dists.ravel(), idxs.ravel()

    inliers = []
    for ia, (d, ib) in enumerate(zip(dists, idxs)):
        if d > dist_thresh:
            continue
        if use_type and int(mins_a[ia,2]) != int(types_b[ib]):
            continue
        ang_diff_val = abs(angle_diff(orients_a[ia] + theta, orients_b[ib]))
        if ang_diff_val > orient_thresh:
            continue
        gauss = math.exp(-(d**2)/(2*(dist_thresh/2.0)**2))
        orient_factor = math.exp(-(ang_diff_val**2)/(2*(orient_thresh/2.0)**2))
        combined = gauss * weights_a[ia] * weights_b[ib] * orient_factor
        inliers.append((ia, int(ib), combined))

    return inliers, len(inliers)

def ransac_worker(args):
    mins_a, mins_b, dist_thresh, orient_thresh, min_inliers, use_type, weights_a, weights_b, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    nA, nB = mins_a.shape[0], mins_b.shape[0]
    idxs_a, idxs_b = np.arange(nA), np.arange(nB)

    best_local = {"theta":0.0, "t":np.array([0.0,0.0]), "inliers":[], "score":0.0}

    # Sample a pair
    ia = int(rng.choice(idxs_a, p=weights_a/weights_a.sum()))
    if use_type:
        candidates_b = idxs_b[mins_b[:,2] == mins_a[ia,2]]
        if candidates_b.size == 0:
            ib = int(rng.choice(idxs_b, p=weights_b/weights_b.sum()))
        else:
            pb = weights_b[candidates_b]; pb /= pb.sum()
            ib = int(rng.choice(candidates_b, p=pb))
    else:
        ib = int(rng.choice(idxs_b, p=weights_b/weights_b.sum()))

    # Estimate transform
    theta, t = estimate_transform_rigid_by_pair(mins_a[ia,:2], mins_b[ib,:2],
                                                mins_a[ia,3], mins_b[ib,3])
    inliers, n_in = match_with_transform(mins_a, mins_b, theta, t,
                                         dist_thresh, orient_thresh,
                                         weights_a, weights_b, use_type)

    if n_in >= min_inliers:
        total_score = sum([c for (_,_,c) in inliers])
        normalized = total_score / max(1.0, min(nA, nB))
        best_local = {"theta": theta, "t": t, "inliers": inliers, "score": normalized}

    return best_local

def ransac_align_and_match_parallel(mins_a, mins_b, dist_thresh,
                                   orient_thresh=math.radians(20.0),
                                   max_iter=300, min_inliers=6,
                                   use_type=True, stop_inlier_ratio=0.6):
    if mins_a.shape[0]==0 or mins_b.shape[0]==0:
        return {"theta":0.0,"t":np.array([0.0,0.0]),"inliers":[],"score":0.0,"inlier_ratio":0.0}

    weights_a = np.array([compute_descriptor_weight(m) for m in mins_a])
    weights_b = np.array([compute_descriptor_weight(m) for m in mins_b])
    best = {"theta":0.0,"t":np.array([0.0,0.0]),"inliers":[],"score":0.0}

    args_list = [(mins_a, mins_b, dist_thresh, orient_thresh, min_inliers, use_type, weights_a, weights_b, seed)
                 for seed in range(max_iter)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ransac_worker, args) for args in args_list]
        for fut in as_completed(futures):
            result = fut.result()
            if result["score"] > best["score"]:
                best = result
                if len(best["inliers"]) / max(1, min(len(mins_a), len(mins_b))) >= stop_inlier_ratio:
                    break

    # Refine transform with inliers if sufficient
    if len(best["inliers"]) >= 3:
        idxsA = np.array([i for i,_,_ in best["inliers"]])
        idxsB = np.array([j for _,j,_ in best["inliers"]])
        Pa, Pb = mins_a[idxsA,:2], mins_b[idxsB,:2]
        ca, cb = Pa.mean(axis=0), Pb.mean(axis=0)
        A_centered, B_centered = Pa - ca, Pb - cb
        H = A_centered.T.dot(B_centered)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T.dot(U.T)
        theta_ref = math.atan2(R[1,0], R[0,0])
        t_ref = cb - rotate_points(ca.reshape(1,2), theta_ref).reshape(2)
        inliers, n_in = match_with_transform(mins_a, mins_b, theta_ref, t_ref,
                                             dist_thresh, orient_thresh, weights_a, weights_b, use_type)
        total_score = sum([c for (_,_,c) in inliers])
        normalized = total_score / max(1.0, min(len(mins_a), len(mins_b)))
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
                        stop_inlier_ratio=0.6,
                        cross_check=True) -> Dict:

    if mins_a is None or mins_b is None:
        return {"final_score":0.0,"inlier_ratio":0.0,"matches":[],"theta":0.0,"t":np.array([0.0,0.0])}

    A, B = np.array(mins_a, copy=True), np.array(mins_b, copy=True)
    orient_thresh = math.radians(orient_thresh_deg)
    best = ransac_align_and_match_parallel(A, B, dist_thresh, orient_thresh,
                                           max_iter=ransac_iter, min_inliers=min_inliers,
                                           use_type=use_type, stop_inlier_ratio=stop_inlier_ratio)

    matches = best.get("inliers", [])

    if cross_check and matches:
        transformed_A = apply_rigid_transform(A[:,:2], best["theta"], best["t"])
        treeA = KDTree(transformed_A)
        idx_b_to_a = treeA.query(B[:,:2], k=1)[1].ravel()
        matches = [(ia, ib, sc) for ia, ib, sc in matches if idx_b_to_a[ib] == ia]
        best["inliers"] = matches

    return {"final_score":float(best["score"]),
            "inlier_ratio":float(best.get("inlier_ratio",0.0)),
            "matches":matches,
            "theta":float(best.get("theta",0.0)),
            "t":best.get("t",np.array([0.0,0.0]))}
