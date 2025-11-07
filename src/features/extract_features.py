import os
import cv2
import numpy as np
from typing import List, Dict, Optional
import json
from scipy.spatial import cKDTree
from src.features.utils import compute_orientation_map
from config import config as cfg

# ============================================================
# 1. Stima orientamento locale di una minutia
# ============================================================
def estimate_minutia_orientation(skel: np.ndarray, x: int, y: int, window: Optional[int] = None) -> float:
    if window is None:
        window = getattr(cfg, "ORIENTATION_WINDOW", 15)

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
def nms_min_distance(minutiae: List[Dict], min_dist: Optional[float] = None) -> List[Dict]:
    if min_dist is None:
        min_dist = getattr(cfg, "MIN_DISTANCE", 6.0)

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


# ============================================================
# 3. Estrazione minutiae reali dallo scheletro
# ============================================================
def extract_minutiae_from_skeleton(skel: np.ndarray) -> List[Dict]:
    """
    Estrae terminazioni e biforcazioni da un'immagine scheletrizzata.
    Restituisce una lista di dizionari {"x": int, "y": int, "type": str}.
    """
    minutiae = []
    sk = (skel > 0).astype(np.uint8)
    if sk.sum() == 0:
        return minutiae

    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)

    neighbor_count = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero(sk)

    for y, x in zip(ys, xs):
        count = neighbor_count[y, x] - 10
        if count == 1:
            minutiae.append({"x": x, "y": y, "type": "ending"})
        elif count == 3:
            minutiae.append({"x": x, "y": y, "type": "bifurcation"})
    return minutiae


# ============================================================
# 4. Post-processing completo delle minutiae
# ============================================================
def postprocess_minutiae(
    minutiae: List[Dict],
    skel: np.ndarray,
    gray: Optional[np.ndarray] = None,
    params: Optional[Dict] = None
) -> List[Dict]:

    if not minutiae or skel is None or skel.sum() == 0:
        return []

    params = params or {}
    qwin = params.get("quality_window", getattr(cfg, "QUALITY_WINDOW", 25))
    qth = params.get("quality_threshold", getattr(cfg, "QUALITY_THRESHOLD", 0.1))
    coh_th = params.get("coherence_threshold", getattr(cfg, "COHERENCE_THRESHOLD", 0.15))
    min_dist = params.get("min_distance", getattr(cfg, "MIN_DISTANCE", 6.0))
    ori_win = params.get("orientation_window", getattr(cfg, "ORIENTATION_WINDOW", 15))
    margin = params.get("margin", getattr(cfg, "MARGIN", 35))

    sk_bin = (skel > 0).astype(np.uint8)
    h, w = sk_bin.shape

    density = cv2.blur(sk_bin.astype(np.float32), (qwin, qwin))
    density /= (density.max() + 1e-6)

    orient, coherence = compute_orientation_map(gray if gray is not None else sk_bin)
    coherence = np.clip(coherence, 0, 1)

    enriched = []
    for m in minutiae:
        x, y = int(m["x"]), int(m["y"])

        if x < margin or x > (w - margin) or y < margin or y > (h - margin):
            continue

        local_coh = float(coherence[y, x])
        local_density = float(density[y, x])

        if local_density < qth or local_coh < coh_th:
            continue

        ang = estimate_minutia_orientation(sk_bin, x, y, ori_win)
        if np.isnan(ang):
            continue

        q = 0.6 * local_coh + 0.4 * local_density
        m.update({
            "orientation": ang,
            "quality": q,
            "coherence": local_coh
        })
        enriched.append(m)

    return nms_min_distance(enriched, min_dist)


# ============================================================
# 5. Elaborazione singolo sample
# ============================================================
def process_sample(debug_dir: str, output_dir: str):
    sample_name = os.path.basename(debug_dir)
    print(f"\nElaborazione di: {sample_name}")

    skel_path = os.path.join(debug_dir, f"{sample_name}_skeleton.png")
    gray_path = os.path.join(debug_dir, f"{sample_name}_segmented.png")

    if not os.path.exists(skel_path):
        print(f"Skeleton non trovato: {skel_path}")
        return

    if not os.path.exists(gray_path):
        print(f"Segmentata non trovata, uso skeleton come fallback.")
        gray_path = skel_path

    skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

    # Estrazione minutiae reali
    test_minutiae = extract_minutiae_from_skeleton(skel)
    print(f"Minutiae trovate sullo scheletro: {len(test_minutiae)}")

    refined = postprocess_minutiae(test_minutiae, skel, gray)
    print(f"Minutiae finali dopo post-processing: {len(refined)}")

    os.makedirs(output_dir, exist_ok=True)

    img_out_path = os.path.join(output_dir, f"{sample_name}_minutiae_postprocessed.png")
    json_out_path = os.path.join(output_dir, f"{sample_name}_minutiae.json")

    vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    for m in refined:
        x, y = int(m["x"]), int(m["y"])
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(img_out_path, vis)

    for m in refined:
        m["x"], m["y"] = int(m["x"]), int(m["y"])
    with open(json_out_path, "w") as f:
        json.dump(refined, f, indent=2)

    print(f"Risultato salvato: {img_out_path}")
    print(f"Dati minutiae salvati: {json_out_path}")


# ============================================================
# 6. Elaborazione batch
# ============================================================
def main():
    input_base = os.path.join("data", "processed", "debug")
    output_base = os.path.join("data", "features", "minutiae")

    if not os.path.exists(input_base):
        raise FileNotFoundError(f"Cartella non trovata: {input_base}")

    os.makedirs(output_base, exist_ok=True)

    sample_dirs = [
        os.path.join(input_base, d)
        for d in os.listdir(input_base)
        if os.path.isdir(os.path.join(input_base, d))
    ]

    if not sample_dirs:
        print("Nessuna sottocartella trovata.")
        return

    print(f"Trovate {len(sample_dirs)} impronte da elaborare.")

    for debug_dir in sample_dirs:
        sample_name = os.path.basename(debug_dir)
        output_dir = os.path.join(output_base, sample_name)
        try:
            process_sample(debug_dir, output_dir)
        except Exception as e:
            print(f"Errore su {sample_name}: {e}")


if __name__ == "__main__":
    main()
