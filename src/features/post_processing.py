import numpy as np
import cv2
from typing import List, Dict, Optional
from scipy.spatial import cKDTree
from src.preprocessing.orientation import compute_orientation_map

# ------------------------------------------------------------
# NMS adattivo basato sulla densità locale
# ------------------------------------------------------------
def nms_adaptive(minutiae: List[Dict], density_map: np.ndarray, base_dist: float = 8.0) -> List[Dict]:
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
        y, x = minutiae[i]["y"], minutiae[i]["x"]
        local_density = density_map[y, x]
        adaptive_radius = base_dist / (0.5 + local_density)
        keep_mask[i] = True
        neighbors = tree.query_ball_point(coords[i], r=adaptive_radius)
        for j in neighbors:
            if j != i:
                keep_mask[j] = False

    return [m for i, m in enumerate(minutiae) if keep_mask[i]]

# ------------------------------------------------------------
# Rimozione ridondanza orientamento adattiva
# ------------------------------------------------------------
def remove_redundant_oriented_adaptive(minutiae: List[Dict],
                                       density_map: np.ndarray,
                                       base_radius: float = 20.0,
                                       angle_thresh: float = np.deg2rad(30)) -> List[Dict]:
    if not minutiae:
        return []

    coords = np.array([[m["x"], m["y"]] for m in minutiae])
    tree = cKDTree(coords)
    to_remove = set()

    for i, m1 in enumerate(minutiae):
        if i in to_remove:
            continue
        y, x = m1["y"], m1["x"]
        local_density = density_map[y, x]
        radius = base_radius * (1.0 + (1.0 - m1.get("quality", 1.0))) / (0.5 + local_density)
        for j in tree.query_ball_point(coords[i], r=radius):
            if j <= i or j in to_remove:
                continue
            ang_diff = abs(np.arctan2(
                np.sin(m1["orientation"] - minutiae[j]["orientation"]),
                np.cos(m1["orientation"] - minutiae[j]["orientation"])
            ))
            if ang_diff < angle_thresh:
                to_remove.add(i if float(m1.get("quality", 1.0)) < float(minutiae[j].get("quality", 1.0)) else j)

    return [m for k, m in enumerate(minutiae) if k not in to_remove]

# ------------------------------------------------------------
# Post-processing ottimizzato
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
    max_m = params.get("max_minutiae", 80)
    patch_r = params.get("patch_radius", 15)

    sk_bin = (skel > 0).astype(np.uint8)
    h, w = sk_bin.shape

    # mappa densità locale
    density = cv2.blur(sk_bin.astype(np.float32), (qwin, qwin))
    density /= (density.max() + 1e-6)

    # calcola orientamento e coerenza
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

        # patch per stabilità angolare
        patch = orient[max(0, y - patch_r):min(h, y + patch_r),
                       max(0, x - patch_r):min(w, x + patch_r)]
        angular_stability = float(np.exp(-3.0 * np.std(patch))) if patch.size > 0 else 0.0

        # posizione centrale
        center_bonus = 1.0 - 0.5 * ((abs(x - w / 2) / (w / 2)) ** 2 + (abs(y - h / 2) / (h / 2)) ** 2)

        # intensità locale
        local_intensity = float(sk_bin[y, x])

        # score combinato finale
        score = (0.5 * local_coh + 0.25 * local_density + 0.1 * angular_stability + 0.1 * local_intensity) * center_bonus

        m.update({
            "orientation": ang,
            "quality": score,
            "coherence": local_coh,
            "angular_stability": angular_stability,
        })
        enriched.append(m)

    # NMS adattivo
    refined = nms_adaptive(enriched, density_map=density, base_dist=min_dist)
    # Rimozione ridondanza orientamento adattiva
    refined = remove_redundant_oriented_adaptive(refined, density_map=density, base_radius=20.0, angle_thresh=np.deg2rad(30))
    # Ordinamento finale e taglio al massimo
    refined = sorted(refined, key=lambda m: float(m["quality"]), reverse=True)[:max_m]

    return refined
