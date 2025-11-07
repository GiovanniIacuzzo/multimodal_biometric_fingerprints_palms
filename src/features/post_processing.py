import numpy as np
import cv2
from typing import List, Dict, Optional
from scipy.spatial import cKDTree
from src.features.utils import compute_orientation_map


# ============================================================
# 1. Stima orientamento locale di una minutia
# ============================================================
def estimate_minutia_orientation(skel: np.ndarray, x: int, y: int, window: int = 15) -> float:
    h, w = skel.shape
    r = window // 2
    x1, x2 = max(0, x - r), min(w, x + r + 1)
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    patch = skel[y1:y2, x1:x2]

    ys, xs = np.nonzero(patch)
    if len(xs) < 10:
        return np.nan

    coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    cov = coords.T @ coords
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]

    angle = np.arctan2(main_dir[1], main_dir[0])
    return float(np.mod(angle + np.pi, np.pi) - np.pi / 2)


# ============================================================
# 2. Non-Maximum Suppression spaziale
# ============================================================
def nms_min_distance(minutiae: List[Dict], min_dist: float = 8.0) -> List[Dict]:
    if not minutiae:
        return []

    coords = np.array([[m["x"], m["y"]] for m in minutiae])
    qualities = np.array([m.get("quality", 1.0) for m in minutiae])
    order = np.argsort(-qualities)  # decrescente

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


# ============================================================
# 3. Post-processing completo delle minutiae
# ============================================================
def postprocess_minutiae(
    minutiae: List[Dict],
    skel: np.ndarray,
    gray: Optional[np.ndarray] = None,
    params: Optional[Dict] = None
) -> List[Dict]:
    if not minutiae or skel is None:
        return []

    # --- Parametri di default ---
    params = params or {}
    qwin = params.get("quality_window", 25)
    qth = params.get("quality_threshold", 0.1)
    coh_th = params.get("coherence_threshold", 0.15)
    min_dist = params.get("min_distance", 6.0)
    ori_win = params.get("orientation_window", 15)
    margin = params.get("margin", 35)

    sk_bin = (skel > 0).astype(np.uint8)
    h, w = sk_bin.shape

    # --- Mappa di densità normalizzata ---
    density = cv2.blur(sk_bin.astype(np.float32), (qwin, qwin))
    density /= (density.max() + 1e-6)

    # --- Mappa di orientazione e coerenza ---
    orient, coherence = compute_orientation_map(gray if gray is not None else sk_bin)
    coherence = np.clip(coherence, 0, 1)

    enriched = []
    for m in minutiae:
        x, y = int(m["x"]), int(m["y"])

        # Bordo immagine
        if x < margin or x > (w - margin) or y < margin or y > (h - margin):
            continue

        local_coh = float(coherence[y, x])
        local_density = float(density[y, x])

        # Filtra per bassa qualità locale
        if local_density < qth or local_coh < coh_th:
            continue

        # Orientamento stimato
        ang = estimate_minutia_orientation(sk_bin, x, y, ori_win)
        if np.isnan(ang):
            continue

        # Qualità complessiva (peso tra coerenza e densità)
        q = 0.6 * local_coh + 0.4 * local_density
        m.update({
            "orientation": ang,
            "quality": q,
            "coherence": local_coh
        })
        enriched.append(m)

    # --- Non-Maximum Suppression ---
    refined = nms_min_distance(enriched, min_dist)

    return refined
