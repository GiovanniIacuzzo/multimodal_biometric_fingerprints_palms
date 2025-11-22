import numpy as np
import math
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.matching.utils import rotate_points, angle_diff
import logging

logger = logging.getLogger(__name__)

def compute_descriptor_weight(m):
    # ending=0, bifurcation=1 → la biforcazione pesa di più
    type_bonus = 1.25 if int(m[2]) == 1 else 1.0

    q = float(m[4]) if len(m) > 4 else 0.0
    coh = float(m[5]) if len(m) > 5 else 0.0
    angs = float(m[6]) if len(m) > 6 else 0.0

    # normalizzazione leggera
    base = (0.5 * q + 0.3 * coh + 0.2 * angs)

    return float(np.clip(type_bonus * base, 0.05, 2.0))

def estimate_transform_rigid_by_pair(p_a, p_b, orient_a, orient_b):
    theta = angle_diff(orient_b, orient_a)
    rotated = rotate_points(p_a.reshape(1,2), theta).reshape(2)
    t = p_b - rotated
    return theta, t

def apply_rigid_transform(points_xy, theta, t):
    return rotate_points(points_xy, theta) + t

def match_with_transform(mins_a, mins_b, theta, t,
                         dist_thresh, orient_thresh,
                         weights_a, weights_b, use_type):
    if mins_a.shape[0] == 0 or mins_b.shape[0] == 0:
        return [], 0

    pts_a = mins_a[:, :2]
    pts_b = mins_b[:, :2]
    types_a = mins_a[:, 2]
    types_b = mins_b[:, 2]
    orients_a = mins_a[:, 3]
    orients_b = mins_b[:, 3]

    transformed = apply_rigid_transform(pts_a, theta, t)
    tree_b = KDTree(pts_b)
    dists, idxs = tree_b.query(transformed, k=1)
    dists, idxs = dists.ravel(), idxs.ravel()

    inliers = []

    for ia, (d, ib) in enumerate(zip(dists, idxs)):
        if d > dist_thresh:
            continue

        if use_type and types_a[ia] != types_b[ib]:
            continue

        ang_err = abs(angle_diff(orients_a[ia] + theta, orients_b[ib]))
        if ang_err > orient_thresh:
            continue

        # peso combinato robusto
        spatial = math.exp(-(d**2) / (2 * (dist_thresh/2)**2))
        orient_factor = math.exp(-(ang_err**2) / (2 * (orient_thresh/2)**2))

        score = spatial * orient_factor * weights_a[ia] * weights_b[ib]
        inliers.append((ia, ib, float(score)))

    return inliers, len(inliers)

def ransac_worker(args):
    mins_a, mins_b, dist_thresh, orient_thresh, min_inliers, use_type, wA, wB, seed = args

    rng = np.random.default_rng(seed)

    # === EARLY REJECT 1: troppi pochi punti
    if mins_a.shape[0] < 8 or mins_b.shape[0] < 8:
        return {"score": 0.0, "inliers": []}

    # === EARLY REJECT 2: distribuzione troppo diversa (anti impostori)
    std_a = mins_a[:, :2].std(0)
    std_b = mins_b[:, :2].std(0)
    if np.linalg.norm(std_a - std_b) > 20:
        return {"score": 0.0, "inliers": []}

    idxsA = np.arange(mins_a.shape[0])

    # === campiona punto ponderato
    pA = rng.choice(idxsA, p=wA/np.sum(wA))

    # === trova candidati coerenti in B
    same_type_idx = np.where(mins_b[:,2] == mins_a[pA,2])[0]
    if len(same_type_idx) == 0:
        return {"score": 0.0, "inliers": []}

    pB = rng.choice(same_type_idx, p=wB[same_type_idx]/np.sum(wB[same_type_idx]))

    # === stima trasformazione rigida
    theta, t = estimate_transform_rigid_by_pair(
        mins_a[pA,:2], mins_b[pB,:2], mins_a[pA,3], mins_b[pB,3]
    )

    # === match con la trasformazione
    inliers, n = match_with_transform(
        mins_a, mins_b, theta, t,
        dist_thresh, orient_thresh,
        wA, wB, use_type
    )

    # === filtro robusto: almeno X inlier veri
    if n < min_inliers:
        return {"score": 0.0, "inliers": []}

    # === nuovo scoring più severo
    weighted = sum([c for (_,_,c) in inliers])
    possible = min(np.sum(wA), np.sum(wB))

    # punizione quadratica (impostori → punteggio crolla)
    score = (weighted / possible)**2

    return {"theta": theta, "t": t, "inliers": inliers, "score": float(score)}

def ransac_align_and_match_parallel(mins_a, mins_b,
                                   dist_thresh,
                                   orient_thresh,
                                   max_iter,
                                   min_inliers,
                                   use_type,
                                   stop_inlier_ratio,
                                   thread_workers):

    # === early reject se vuoto
    if len(mins_a) == 0 or len(mins_b) == 0:
        return {"score": 0.0, "inliers": []}

    # === calcolo pesi
    wA = np.array([compute_descriptor_weight(m) for m in mins_a])
    wB = np.array([compute_descriptor_weight(m) for m in mins_b])

    base_seed = 42
    args_list = [
        (mins_a, mins_b, dist_thresh, orient_thresh, min_inliers,
         use_type, wA, wB, base_seed+i)
        for i in range(max_iter)
    ]

    best = {"score": 0.0, "inliers": []}

    # === parallel RANSAC
    with ThreadPoolExecutor(max_workers=thread_workers) as ex:
        futures = [ex.submit(ransac_worker, a) for a in args_list]
        for f in as_completed(futures):
            r = f.result()

            if r["score"] > best["score"]:
                best = r

            # early stop se trasformazione già molto buona
            if len(r.get("inliers", [])) >= stop_inlier_ratio * min(len(mins_a), len(mins_b)):
                best = r
                break

    # === nessun modello accettabile
    if best["score"] <= 0:
        return best

    # === raffinamento finale con SVD
    idxA = np.array([i for (i,_,_) in best["inliers"]])
    idxB = np.array([j for (_,j,_) in best["inliers"]])
    Pa = mins_a[idxA,:2]
    Pb = mins_b[idxB,:2]

    ca, cb = Pa.mean(0), Pb.mean(0)
    A = Pa - ca
    B = Pb - cb

    H = A.T @ B
    U,_,Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    theta = math.atan2(R[1,0], R[0,0])
    t = cb - rotate_points(ca.reshape(1,2), theta).reshape(2)

    # === ricalcolo inlier con trasformazione raffinata
    inliers, _ = match_with_transform(
        mins_a, mins_b, theta, t,
        dist_thresh, orient_thresh,
        wA, wB, use_type
    )

    # === nuovo score severo
    weighted = sum([c for (_,_,c) in inliers])
    possible = min(np.sum(wA), np.sum(wB))
    score = (weighted / possible)**2

    # === controllo geometrico finale (anti-impostori)
    if len(inliers) >= 8:
        Pa = mins_a[[i for (i,_,_) in inliers], :2]
        Pb = mins_b[[j for (_,j,_) in inliers], :2]

        dA = np.linalg.norm(Pa - Pa.mean(0), axis=1).mean()
        dB = np.linalg.norm(Pb - Pb.mean(0), axis=1).mean()

        if abs(dA - dB) > 10:
            # geometria incoerente → quasi sicuramente impostore
            return {"score": 0.0, "inliers": []}

    return {"theta": theta, "t": t, "inliers": inliers, "score": float(score)}

def match_minutiae_pair(
    mins_a, mins_b,
    dist_thresh=10.0,
    orient_thresh_deg=12.0,
    use_type=True,
    ransac_iter=300,
    min_inliers=8,
    stop_inlier_ratio=0.25,
    cross_check=True,
    thread_workers=4,
    debug=False
):

    if mins_a is None or mins_b is None:
        return {"final_score": 0.0, "inlier_ratio": 0.0, "matches": []}

    A = np.array(mins_a)
    B = np.array(mins_b)
    orient_thresh = math.radians(orient_thresh_deg)

    best = ransac_align_and_match_parallel(
        A, B,
        dist_thresh, orient_thresh,
        ransac_iter,
        min_inliers,
        use_type,
        stop_inlier_ratio,
        thread_workers
    )

    inliers = best.get("inliers", [])

    # optional cross-check migliorato
    if cross_check and len(inliers) > 0:
        transformed_A = apply_rigid_transform(A[:,:2], best["theta"], best["t"])
        treeA = KDTree(transformed_A)
        idx_b_to_a = treeA.query(B[:,:2], k=1)[1].ravel()
        inliers = [(i,j,s) for (i,j,s) in inliers if idx_b_to_a[j] == i]

    final_score = float(np.clip(best.get("score", 0.0), 0.0, 1.0))
    inlier_ratio = len(inliers) / max(1, min(len(A),len(B)))

    return {
        "final_score": final_score,
        "inlier_ratio": inlier_ratio,
        "matches": inliers,
        "theta": best.get("theta", 0.0),
        "t": best.get("t", np.array([0.0,0.0]))
    }
