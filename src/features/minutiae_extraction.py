"""
minutiae_extraction.py
----------------------
Modulo per l'estrazione automatica delle minutiae (terminazioni e biforcazioni)
da immagini skeletonizzate di impronte digitali.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import rotate
from scripts.config import PROCESSED_DIR, FEATURES_DIR, DATASET_DIR

# ==========================
# 1. CROSSING NUMBER METHOD
# ==========================

def crossing_number(neighborhood: np.ndarray) -> int:
    p = neighborhood.flatten()
    sequence = [p[1], p[2], p[5], p[8], p[7], p[6], p[3], p[0], p[1]]
    cn = 0.5 * np.sum(np.abs(np.diff(sequence)))
    return int(cn)

# ==========================
# 2. MINUTIAE DETECTION
# ==========================

def extract_minutiae(skeleton: np.ndarray, min_distance: int = 5) -> list:
    minutiae = []
    rows, cols = skeleton.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 255:
                window = skeleton[i-1:i+2, j-1:j+2] // 255
                cn = crossing_number(window)
                if cn == 1:
                    minutiae.append({"x": j, "y": i, "type": "ending"})
                elif cn == 3:
                    minutiae.append({"x": j, "y": i, "type": "bifurcation"})

    filtered = []
    for m in minutiae:
        too_close = any(np.hypot(m["x"] - n["x"], m["y"] - n["y"]) < min_distance for n in filtered)
        if not too_close:
            filtered.append(m)
    return filtered

# ==========================
# 3. ORIENTAZIONE LOCALE
# ==========================

def compute_local_orientation(skeleton: np.ndarray, x: int, y: int, window_size: int = 9) -> float:
    half = window_size // 2
    x_min, x_max = max(0, x - half), min(skeleton.shape[1], x + half)
    y_min, y_max = max(0, y - half), min(skeleton.shape[0], y + half)
    region = (skeleton[y_min:y_max, x_min:x_max] == 255).astype(np.uint8)
    coords = np.column_stack(np.nonzero(region))
    if len(coords) < 2:
        return 0.0
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    return float(angle)

# ==========================
# 4. PIPELINE COMPLETA
# ==========================

def process_skeleton_image(skeleton_path: str, output_json: str):
    skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    minutiae = extract_minutiae(skeleton)
    for m in minutiae:
        m["orientation"] = compute_local_orientation(skeleton, m["x"], m["y"])
    with open(output_json, "w") as f:
        json.dump(minutiae, f, indent=2)

# ==========================
# 5. MAIN
# ==========================

def main():
    """Esegue l’estrazione di tutte le minutiae dal dataset."""
    catalog_path = os.path.join(DATASET_DIR, "catalog.csv")
    output_dir = os.path.join(FEATURES_DIR, "minutiae")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(catalog_path)

    print(f"🧬 Estrazione minutiae da {len(df)} immagini...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_name = os.path.splitext(os.path.basename(row["path"]))[0]
        skeleton_path = os.path.join(PROCESSED_DIR, base_name, "skeleton.png")
        output_json = os.path.join(output_dir, f"{base_name}_minutiae.json")
        if os.path.exists(skeleton_path):
            process_skeleton_image(skeleton_path, output_json)

    print(f"\n✅ Estrazione completata! File salvati in: {output_dir}")

if __name__ == "__main__":
    main()
