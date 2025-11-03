"""
minutiae_extraction.py
------------------------------------------
Estrazione minutiae robusta e performante da immagini skeletonizzate.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.config import PROCESSED_DIR, FEATURES_DIR, DATASET_DIR


# ==========================
# CROSSING NUMBER
# ==========================
def crossing_number(neighborhood: np.ndarray) -> int:
    """
    Calcola il Crossing Number su un intorno 3x3 binarizzato.
    """
    p = neighborhood.flatten()
    seq = [p[1], p[2], p[5], p[8], p[7], p[6], p[3], p[0], p[1]]
    cn = 0.5 * np.sum(np.abs(np.diff(seq)))
    return int(cn)


# ==========================
# MINUTIAE EXTRACTION
# ==========================
def extract_minutiae(skeleton: np.ndarray, min_distance: int = 4, border_margin: int = 4) -> list:
    """
    Estrae terminazioni e biforcazioni con filtri anti-rumore e distanza minima.
    """
    # Binarizzazione sicura
    skeleton = (skeleton > 0).astype(np.uint8)

    minutiae = []
    rows, cols = skeleton.shape

    # Scansione pixel
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                window = skeleton[i - 1:i + 2, j - 1:j + 2]
                cn = crossing_number(window)
                if cn == 1:
                    minutiae.append({"x": j, "y": i, "type": "ending"})
                elif cn == 3:
                    minutiae.append({"x": j, "y": i, "type": "bifurcation"})

    # Rimozione bordi e troppo vicini
    filtered = []
    for m in minutiae:
        if (
            m["x"] < border_margin or m["x"] >= cols - border_margin or
            m["y"] < border_margin or m["y"] >= rows - border_margin
        ):
            continue
        if all(np.hypot(m["x"] - n["x"], m["y"] - n["y"]) >= min_distance for n in filtered):
            filtered.append(m)

    return filtered


# ==========================
# LOCAL ORIENTATION
# ==========================
def compute_local_orientation(skeleton: np.ndarray, x: int, y: int, window_size: int = 9) -> float:
    """
    Calcola orientazione locale tramite struttura tensoriale.
    """
    half = window_size // 2
    x_min, x_max = max(0, x - half), min(skeleton.shape[1], x + half)
    y_min, y_max = max(0, y - half), min(skeleton.shape[0], y + half)

    region = (skeleton[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
    if np.count_nonzero(region) < 3:
        return 0.0

    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    vx = 2 * np.sum(gx * gy)
    vy = np.sum(gx ** 2 - gy ** 2)
    theta = 0.5 * np.arctan2(vx, vy)
    return float(theta)


# ==========================
# PROCESS SINGLE IMAGE
# ==========================
def process_skeleton_image(skeleton_path: str, output_json: str):
    skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        print(f"⚠️ Impossibile leggere {skeleton_path}")
        return

    # Filtro leggero per ridurre rumore
    skeleton = cv2.medianBlur(skeleton, 3)
    minutiae = extract_minutiae(skeleton)

    if not minutiae:
        print(f"⚠️ Nessuna minutia trovata in {skeleton_path}")

    for m in minutiae:
        m["orientation"] = compute_local_orientation(skeleton, m["x"], m["y"])

    with open(output_json, "w") as f:
        json.dump(minutiae, f, indent=2)


# ==========================
# MAIN
# ==========================
def main():
    catalog_path = os.path.join(DATASET_DIR, "catalog.csv")
    output_dir = os.path.join(FEATURES_DIR, "minutiae")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(catalog_path)

    print(f"🧬 Estrazione robusta minutiae da {len(df)} immagini...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_name = os.path.splitext(os.path.basename(row["path"]))[0]
        skeleton_path = os.path.join(PROCESSED_DIR, base_name, "skeleton.png")
        output_json = os.path.join(output_dir, f"{base_name}_minutiae.json")
        if os.path.exists(skeleton_path):
            process_skeleton_image(skeleton_path, output_json)

    print(f"\n✅ Estrazione completata! File salvati in: {output_dir}")


if __name__ == "__main__":
    main()
