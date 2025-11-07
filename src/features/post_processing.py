import numpy as np
import cv2
from typing import List, Dict
from scipy.spatial import cKDTree
from src.features.utils import compute_orientation_map

def estimate_minutia_orientation(skel: np.ndarray, x: int, y: int, window=11) -> float:
    h, w = skel.shape
    r = window // 2
    patch = skel[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
    ys, xs = np.nonzero(patch)
    if len(xs) < 3:
        return 0.0
    coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    cov = coords.T @ coords
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.arctan2(eigvecs[1, -1], eigvecs[0, -1])
    return float(np.mod(angle + np.pi/2, np.pi) - np.pi/2)

def nms_min_distance(minutiae: List[Dict], min_dist=8.0) -> List[Dict]:
    if not minutiae:
        return []
    coords = np.array([[m["x"], m["y"]] for m in minutiae])
    qualities = np.array([m.get("quality", 1.0) for m in minutiae])
    order = np.argsort(-qualities)
    keep = np.ones(len(minutiae), bool)
    tree = cKDTree(coords)
    for i in order:
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], min_dist)
        for n in neighbors:
            if n != i:
                keep[n] = False
    return [m for i, m in enumerate(minutiae) if keep[i]]

def postprocess_minutiae(minutiae, skel, gray=None, params=None) -> List[Dict]:
    if not minutiae or skel is None:
        return []
    params = params or {}
    qwin = params.get("quality_window", 25)
    qth = params.get("quality_threshold", 0.05)
    coh_th = params.get("coherence_threshold", 0.05)
    min_dist = params.get("min_distance", 8.0)
    ori_win = params.get("orientation_window", 11)

    sk_bin = (skel > 0).astype(np.uint8)
    density = cv2.blur(sk_bin.astype(np.float32), (qwin, qwin))
    orient, coherence = compute_orientation_map(gray if gray is not None else sk_bin)

    enriched = []
    h, w = sk_bin.shape
    for m in minutiae:
        x, y = int(m["x"]), int(m["y"])
        if not (0 <= x < w and 0 <= y < h):
            continue
        q = float(density[y, x] * (0.6 + 0.4 * coherence[y, x]))
        if q < qth or coherence[y, x] < coh_th:
            continue
        ang = estimate_minutia_orientation(sk_bin, x, y, ori_win)
        m.update({"quality": q, "orientation": ang, "coherence": float(coherence[y, x])})
        enriched.append(m)

    return nms_min_distance(enriched, min_dist)
