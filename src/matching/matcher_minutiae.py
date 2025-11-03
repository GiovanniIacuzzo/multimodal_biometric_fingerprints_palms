"""
matcher_minutiae.py (robust)
----------------------------
Matching geometrico tra due fingerprint basato su minutiae.
Usa allineamento affino (RANSAC) e punteggio robusto.
"""

import json
import numpy as np
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor


def load_minutiae(json_path: str):
    with open(json_path) as f:
        return json.load(f)


def _coords(minutiae):
    return np.array([[m["x"], m["y"]] for m in minutiae]) if minutiae else np.empty((0, 2))


def align_minutiae_ransac(template, probe, max_trials=100):
    """Allinea probe a template tramite trasformazione affina stimata con RANSAC."""
    t_coords, p_coords = _coords(template), _coords(probe)
    if len(t_coords) < 3 or len(p_coords) < 3:
        return t_coords, p_coords

    # corrispondenze iniziali grezze (KDTree)
    tree = KDTree(t_coords)
    _, idx = tree.query(p_coords, k=1)
    if len(idx) < 3:
        return t_coords, p_coords

    model = RANSACRegressor(min_samples=3, max_trials=max_trials)
    try:
        model.fit(p_coords, t_coords[idx])
        p_aligned = model.predict(p_coords)
        return t_coords, p_aligned
    except Exception:
        return t_coords, p_coords


def minutiae_similarity(template, probe, dist_thresh=15, orient_thresh=np.pi/6):
    """Calcola similarità robusta tra due set di minutiae."""
    if not template or not probe:
        return 0.0

    t_coords, p_coords = align_minutiae_ransac(template, probe)
    if len(t_coords) == 0 or len(p_coords) == 0:
        return 0.0

    tree = KDTree(t_coords)
    matched = 0
    for i, p in enumerate(p_coords):
        dist, j = tree.query(p)
        if dist < dist_thresh:
            ori_diff = abs(template[j]["orientation"] - probe[i]["orientation"])
            if ori_diff < orient_thresh:
                matched += 1

    score = matched / max(len(template), len(probe))
    return float(score)
