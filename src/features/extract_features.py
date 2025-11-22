import os
import json
import logging
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from tqdm import tqdm
from colorama import Fore, Style
from scipy.ndimage import convolve

from src.features.post_processing import postprocess_minutiae

# ====================================================
# LOGGING SETUP
# ====================================================
OUTPUT_DIR_DEFAULT = "dataset/processed/minutiae"
os.makedirs(OUTPUT_DIR_DEFAULT, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR_DEFAULT, "minutiae_extraction.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ====================================================
# UTILITY
# ====================================================
def console_step(title: str):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

def clean_skeleton(skel: np.ndarray) -> np.ndarray:
    return (skel > 127).astype(np.uint8)

def extract_minutiae(skel: np.ndarray) -> List[Dict]:
    sk = clean_skeleton(skel)
    kernel = np.ones((3, 3), dtype=int)
    neigh_count = convolve(sk, kernel, mode='constant', cval=0)

    ys, xs = np.where(sk == 1)
    minutiae = []

    for y, x in zip(ys, xs):
        if y < 1 or x < 1 or y > sk.shape[0] - 2 or x > sk.shape[1] - 2:
            continue

        P = [
            sk[y, x+1], sk[y-1, x+1], sk[y-1, x],
            sk[y-1, x-1], sk[y, x-1], sk[y+1, x-1],
            sk[y+1, x], sk[y+1, x+1]
        ]
        CN = sum(abs(int(P[i]) - int(P[(i+1)%8])) for i in range(8)) // 2

        if CN == 1:
            mtype = "ending"
        elif CN == 3:
            mtype = "bifurcation"
        else:
            continue

        minutiae.append({"x": int(x), "y": int(y), "type": mtype})

    return minutiae

# ====================================================
# PROCESS SINGLE IMAGE
# ====================================================
def process_image(filename, cluster_dir, out_dir, params):
    # Process ONLY skeleton images
    if not filename.endswith("_skeleton.jpg"):
        return

    sample_name = filename.replace("_skeleton.jpg", "")
    skel_path = os.path.join(cluster_dir, filename)

    try:
        skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
        if skel is None:
            logging.error(f"Immagine skeleton corrotta: {skel_path}")
            return

        t0 = time.time()
        raw = extract_minutiae(skel)

        try:
            refined = postprocess_minutiae(raw, skel, skel, params)
        except Exception as e:
            logging.error(f"postprocess_minutiae error on {filename}: {e}")
            refined = []

        # Save visualization
        vis = cv2.cvtColor((skel > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        for m in refined:
            color = (0, 0, 255) if m["type"] == "ending" else (0, 255, 0)
            cv2.circle(vis, (m["x"], m["y"]), 3, color, -1)

        cv2.imwrite(os.path.join(out_dir, f"{sample_name}_minutiae.jpg"), vis)
        with open(os.path.join(out_dir, f"{sample_name}_minutiae.json"), "w") as f:
            json.dump(refined, f, indent=2)

    except Exception as e:
        logging.error(f"Errore elaborando {filename}: {e}")

# ====================================================
# PROCESS CLUSTER DIRECTORY
# ====================================================
def process_cluster_dir(cluster_dir: str, output_base: str, params=None, max_workers=None):

    # Filter ONLY skeleton images
    skeleton_files = [
        f for f in os.listdir(cluster_dir)
        if f.lower().endswith("_skeleton.jpg")
    ]

    if not skeleton_files:
        return

    cluster_name = os.path.basename(cluster_dir)
    out_dir = os.path.join(output_base, cluster_name)
    os.makedirs(out_dir, exist_ok=True)

    console_step(f"Cluster {cluster_name}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(lambda f: process_image(f, cluster_dir, out_dir, params), skeleton_files),
            total=len(skeleton_files),
            desc="Minutiae",
            ncols=90
        ))

# ====================================================
# MAIN
# ====================================================
def main(input_base="dataset/processed/enhanced", output_base=OUTPUT_DIR_DEFAULT, max_workers=None):
    if not os.path.exists(input_base):
        raise FileNotFoundError(f"Input base non trovato: {input_base}")

    cluster_dirs = [
        os.path.join(input_base, d)
        for d in os.listdir(input_base)
        if d.startswith("cluster_")
    ]

    console_step("Rilevamento cluster")
    print(f"→ {len(cluster_dirs)} cluster trovati")
    logging.info(f"Trovati {len(cluster_dirs)} cluster.")

    for cluster in cluster_dirs:
        process_cluster_dir(cluster, output_base, params=None, max_workers=max_workers)

    console_step("Elaborazione completata")
    print(f"{Fore.CYAN}✨ Tutti i cluster processati! ✨{Style.RESET_ALL}")


if __name__ == "__main__":
    main()