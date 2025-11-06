import os
import cv2
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.morphology import thin
from config.config import PROCESSED_DIR, FEATURES_DIR, CATALOG_CSV

# ==========================
# CROSSING NUMBER
# ==========================
def crossing_number(window: np.ndarray) -> int:
    """
    Calcola il Crossing Number in una finestra 3x3
    """
    p = (window > 0).astype(int).flatten()  # Assicurati 0/1
    seq = [p[1], p[2], p[5], p[8], p[7], p[6], p[3], p[0], p[1]]
    cn = 0.5 * np.sum(np.abs(np.diff(seq)))
    return int(cn)

# ==========================
# PULIZIA DELICATA SKELETON
# ==========================
def clean_skeleton(skeleton: np.ndarray) -> np.ndarray:
    skel_bin = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skel_bin, -1, kernel)
    isolated = (skel_bin == 1) & (neighbors <= 1)
    skel_bin[isolated] = 0
    skel_bin = thin(skel_bin).astype(np.uint8)
    return skel_bin * 255

# ==========================
# ESTRAZIONE MINUTIAE
# ==========================
def extract_minutiae(skeleton: np.ndarray, min_distance=1, border_margin=5):
    skel_bin = (skeleton > 0).astype(np.uint8)
    rows, cols = skel_bin.shape
    minutiae = []

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(skel_bin, kernel, iterations=border_margin)

    # --- Scansione finestra 3x3 ---
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skel_bin[i, j] == 1:
                if mask[i, j]:
                    window = skel_bin[i-1:i+2, j-1:j+2]
                    cn = crossing_number(window)
                    if cn == 1:
                        minutiae.append({"x": float(j), "y": float(i), "type": "ending"})
                    elif cn == 3:
                        minutiae.append({"x": float(j), "y": float(i), "type": "bifurcation"})

    filtered = []
    for m in minutiae:
        if all(np.hypot(m["x"] - n["x"], m["y"] - n["y"]) >= min_distance for n in filtered):
            filtered.append(m)

    return filtered

# ==========================
# DEBUG VISIVO
# ==========================
def visualize_minutiae(skeleton: np.ndarray, minutiae: list, output_path: str):
    skel_vis = (skeleton > 0).astype(np.uint8) * 255
    vis = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)

    for m in minutiae:
        color = (0, 0, 255) if m["type"] == "ending" else (0, 255, 0)
        cv2.circle(vis, (int(m["x"]), int(m["y"])), 4, color, 1)
    
    cv2.imwrite(output_path, vis)

# ==========================
# PROCESS SINGLE IMAGE
# ==========================
def process_skeleton(skeleton_path: str, out_json: str, out_vis: str = None):
    skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        print(f"Impossibile leggere {skeleton_path}")
        return

    skeleton_clean = clean_skeleton(skeleton)
    minutiae_raw = extract_minutiae(skeleton_clean)

    minutiae = postprocess_minutiae(
        minutiae_raw,
        skeleton_clean,
        min_distance=15,
        border_margin=20,
        quality_window=25,
        quality_threshold=0.05
    )

    if not minutiae:
        print(f"Nessuna minutia utile trovata in {skeleton_path}")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    minutiae_sorted = sorted(minutiae, key=lambda x: (x["type"], x["y"], x["x"]))

    # --- Salvataggio JSON ---
    with open(out_json, "w") as f:
        json.dump(minutiae_sorted, f, indent=2)

    # --- Salvataggio descrittore Numpy ---
    descriptor = minutiae_to_descriptor(minutiae_sorted)
    out_npy = out_json.replace("_minutiae.json", "_descriptor.npy")
    np.save(out_npy, descriptor)

    # --- Debug visivo ---
    if out_vis:
        os.makedirs(os.path.dirname(out_vis), exist_ok=True)
        visualize_minutiae(skeleton_clean, minutiae_sorted, out_vis)

def postprocess_minutiae(minutiae, skeleton, min_distance=2, border_margin=2, quality_window=10, quality_threshold=0.4):
    filtered = []

    # Rimuovi quelle troppo vicine
    for m in minutiae:
        if all(np.hypot(m["x"] - n["x"], m["y"] - n["y"]) >= min_distance for n in filtered):
            filtered.append(m)

    # Rimuovi quelle troppo vicine ai bordi
    rows, cols = skeleton.shape
    border_filtered = [
        m for m in filtered
        if border_margin <= m["x"] <= cols - border_margin and border_margin <= m["y"] <= rows - border_margin
    ]

    # Rimuovi quelle in aree di bassa qualità (pochi pixel di cresta nel vicinato)
    skel_bin = (skeleton > 0).astype(np.uint8)
    quality_filtered = []
    for m in border_filtered:
        x, y = int(m["x"]), int(m["y"])
        x0, y0 = max(0, x - quality_window), max(0, y - quality_window)
        x1, y1 = min(cols, x + quality_window), min(rows, y + quality_window)
        window = skel_bin[y0:y1, x0:x1]
        local_density = np.mean(window)
        if local_density >= quality_threshold:
            quality_filtered.append(m)

    return quality_filtered

def minutiae_to_descriptor(minutiae):
    """
    Converte lista di minuzie in un descrittore numpy strutturato
    """
    if not minutiae:
        return np.empty((0, 4), dtype=np.float32)

    # Per semplicità: x, y, type, quality (stimata con densità locale)
    desc = []
    for m in minutiae:
        x, y = m["x"], m["y"]
        type_val = 0 if m["type"] == "ending" else 1
        desc.append([x, y, type_val, 1.0])  # qualità = 1.0 per ora
    return np.array(desc, dtype=np.float32)

# ==========================
# MAIN
# ==========================
def main(test_mode=False, debug_vis=True):
    df = pd.read_csv(CATALOG_CSV)
    if test_mode:
        df = df.head(10)
        print(f"Modalità TEST attiva: processate solo {len(df)} immagini.")

    output_dir = os.path.join(FEATURES_DIR, "minutiae")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Estrazione minutiae da {len(df)} immagini...\n")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_name = os.path.splitext(os.path.basename(row["path"]))[0]
        skeleton_path = os.path.join(PROCESSED_DIR, base_name, "skeleton.png")
        out_json = os.path.join(output_dir, f"{base_name}_minutiae.json")
        out_vis = os.path.join(output_dir, f"{base_name}_minutiae_vis.png") if debug_vis else None
        if os.path.exists(skeleton_path):
            process_skeleton(skeleton_path, out_json, out_vis)
        else:
            print(f"Skeleton mancante: {skeleton_path}")

# ==========================
# ENTRY POINT CLI
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrazione minutiae da immagini skeletonizzate")
    parser.add_argument("--test", action="store_true", help="Modalità test: processa solo 10 immagini")
    parser.add_argument("--no-vis", action="store_true", help="Disabilita le immagini di debug")
    args = parser.parse_args()

    main(test_mode=args.test, debug_vis=not args.no_vis)
