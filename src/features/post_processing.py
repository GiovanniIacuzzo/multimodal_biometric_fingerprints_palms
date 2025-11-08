import numpy as np
import cv2
import logging
from typing import List, Dict, Optional
from scipy.spatial import cKDTree
from src.preprocessing.orientation import compute_orientation_map

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------------------------------------------------------
# Non-Maximum Suppression
# ------------------------------------------------------------
def nms_min_distance(minutiae: List[Dict], min_dist: float = 8.0) -> List[Dict]:
    if not minutiae:
        return []

    coords = np.array([[m["x"], m["y"]] for m in minutiae])
    qualities = np.array([m.get("quality", 1.0) for m in minutiae])
    order = np.argsort(-qualities)
    keep_mask = np.zeros(len(minutiae), dtype=bool)
    tree = cKDTree(coords)

    for i in order:
        if keep_mask[i]:
            continue
        keep_mask[i] = True
        neighbors = tree.query_ball_point(coords[i], r=min_dist)
        for j in neighbors:
            if j != i:
                keep_mask[j] = False

    return [m for i, m in enumerate(minutiae) if keep_mask[i]]


# ------------------------------------------------------------
# Filtraggio orientamento
# ------------------------------------------------------------
def remove_redundant_oriented(minutiae: List[Dict],
                              radius: float = 10.0,
                              angle_thresh: float = np.deg2rad(20)) -> List[Dict]:
    if not minutiae:
        return []

    coords = np.array([[m["x"], m["y"]] for m in minutiae])
    tree = cKDTree(coords)
    to_remove = set()

    for i, m1 in enumerate(minutiae):
        if i in to_remove:
            continue
        for j in tree.query_ball_point(coords[i], r=radius):
            if j <= i or j in to_remove:
                continue
            ang_diff = abs(np.arctan2(
                np.sin(m1["orientation"] - minutiae[j]["orientation"]),
                np.cos(m1["orientation"] - minutiae[j]["orientation"])
            ))
            if ang_diff < angle_thresh:
                # Rimuovi la minutia con qualità minore
                to_remove.add(i if float(m1.get("quality", 1.0)) < float(minutiae[j].get("quality", 1.0)) else j)

    return [m for k, m in enumerate(minutiae) if k not in to_remove]


# ------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------
def postprocess_minutiae(minutiae: List[Dict],
                         skel: np.ndarray,
                         gray: Optional[np.ndarray] = None,
                         params: Optional[Dict] = None) -> List[Dict]:
    if not minutiae or skel is None:
        return []

    params = params or {}
    qwin = params.get("quality_window", 25)
    qth = params.get("quality_threshold", 0.15)
    coh_th = params.get("coherence_threshold", 0.2)
    min_dist = params.get("min_distance", 8.0)
    margin = params.get("margin", 30)
    max_m = params.get("max_minutiae", 50)

    sk_bin = (skel > 0).astype(np.uint8)
    h, w = sk_bin.shape

    density = cv2.blur(sk_bin.astype(np.float32), (qwin, qwin))
    density /= (density.max() + 1e-6)
    _, orient, coherence = compute_orientation_map(gray if gray is not None else sk_bin)
    coherence = np.clip(coherence, 0, 1)

    enriched = []
    for m in minutiae:
        x, y = int(m["x"]), int(m["y"])
        if not (margin <= x < w - margin and margin <= y < h - margin):
            continue

        local_coh, local_density = float(coherence[y, x]), float(density[y, x])
        if local_density < qth or local_coh < coh_th:
            continue

        ang = float(orient[y, x])

        r = 10
        patch = orient[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)]
        angular_stability = float(np.exp(-3.0 * np.std(patch))) if patch.size > 0 else 0.0

        margin_penalty = 1.0 - 0.5 * ((abs(x - w / 2) / (w / 2)) ** 2 + (abs(y - h / 2) / (h / 2)) ** 2)
        q = float((0.6 * local_coh + 0.3 * local_density + 0.1 * angular_stability) * margin_penalty)

        m.update({
            "orientation": ang,
            "quality": q,
            "coherence": local_coh,
            "angular_stability": angular_stability,
        })
        enriched.append(m)

    refined = nms_min_distance(enriched, min_dist)
    logging.info(f"Prima filtro orientamento: {len(refined)}")
    refined = remove_redundant_oriented(refined, 20.0, np.deg2rad(30))
    logging.info(f"Dopo filtro orientamento: {len(refined)}")

    refined = sorted(refined, key=lambda m: float(m["quality"]), reverse=True)[:max_m]
    logging.info(f"Totale minutiae finali: {len(refined)}")
    return refined
