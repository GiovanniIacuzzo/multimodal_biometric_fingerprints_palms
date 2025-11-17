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

from src.features.post_processing import postprocess_minutiae
from src.db.database import get_image_id_by_filename, save_minutiae, save_features_summary
from scipy.ndimage import convolve

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
    """Stampa un titolo colorato per una sezione della pipeline."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

def clean_skeleton(skel: np.ndarray) -> np.ndarray:
    """Restituisce uno skeleton binario pulito."""
    return (skel > 127).astype(np.uint8)

def extract_minutiae(skel: np.ndarray) -> List[Dict]:
    """Estrae minutiae usando convoluzione 3x3."""
    sk = clean_skeleton(skel)
    kernel = np.ones((3, 3), dtype=int)
    neigh_count = convolve(sk, kernel, mode='constant', cval=0)
    ys, xs = np.where(sk == 1)
    minutiae = []

    for y, x in zip(ys, xs):
        if y < 1 or x < 1 or y > sk.shape[0]-2 or x > sk.shape[1]-2:
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
def process_image(filename, skel_dir, out_dir, params, db_cache):
    sample_name = os.path.splitext(filename)[0]
    skel_path = os.path.join(skel_dir, filename)
    try:
        skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
        if skel is None:
            logging.error(f"Immagine corrotta: {skel_path}")
            return

        t0 = time.time()
        raw = extract_minutiae(skel)

        try:
            refined = postprocess_minutiae(raw, skel, skel, params)
        except Exception as e:
            logging.error(f"Errore in postprocess_minutiae su {filename}: {e}")
            refined = []

        tproc = time.time() - t0

        image_id = db_cache.get(filename)
        if image_id is None:
            logging.warning(f"'{filename}' non trovato nel DB.")
            return

        if refined:
            save_minutiae(image_id, refined)

        save_features_summary(
            image_id=image_id,
            raw_count=len(raw),
            post_count=len(refined),
            avg_quality=float(np.mean([m.get("quality", 0) for m in refined])) if refined else 0.0,
            avg_coherence=float(np.mean([m.get("coherence", 0) for m in refined])) if refined else 0.0,
            processing_time_sec=tproc,
            params=params or {}
        )

        # --- Output visualization + JSON ---
        out_img = os.path.join(out_dir, f"{sample_name}_minutiae_postprocessed.jpg")
        out_json = os.path.join(out_dir, f"{sample_name}_minutiae.json")
        vis = cv2.cvtColor((skel > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        for m in refined:
            color = (0, 0, 255) if m["type"] == "ending" else (0, 255, 0)
            cv2.circle(vis, (m["x"], m["y"]), 3, color, -1)
        cv2.imwrite(out_img, vis)
        with open(out_json, "w") as f:
            json.dump(refined, f, indent=2)

        logging.info(f"✔ Processata: {filename} ({tproc:.2f}s)")
        print(f"{Fore.GREEN}✔{Style.RESET_ALL} {filename} processata ({tproc:.2f}s)")

    except Exception as e:
        logging.error(f"Errore elaborando {filename}: {e}")

# ====================================================
# PROCESS SINGLE SKELETON DIRECTORY
# ====================================================
def process_skeleton_dir(skel_dir: str, output_base: str, input_base: str, params: Optional[Dict] = None, max_workers=None):
    try:
        jpgs = [f for f in os.listdir(skel_dir) if f.lower().endswith(".jpg")]
    except Exception as e:
        logging.error(f"Impossibile leggere {skel_dir}: {e}")
        return
    if not jpgs:
        return

    rel_cluster_dir = os.path.relpath(os.path.dirname(skel_dir), start=input_base)
    out_dir = os.path.join(output_base, rel_cluster_dir)
    os.makedirs(out_dir, exist_ok=True)

    db_cache = {f: get_image_id_by_filename(f) for f in jpgs}

    console_step(f"Cluster {rel_cluster_dir}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(lambda f: process_image(f, skel_dir, out_dir, params, db_cache), jpgs),
            total=len(jpgs),
            desc="Minutiae",
            ncols=90
        ))

# ====================================================
# MAIN
# ====================================================
def main(input_base="dataset/processed/debug", output_base=OUTPUT_DIR_DEFAULT, max_workers=None):
    if not os.path.exists(input_base):
        raise FileNotFoundError(f"Input base non trovato: {input_base}")

    skeleton_dirs = []
    for root, dirs, files in os.walk(input_base):
        for d in dirs:
            if d.lower() == "skeleton":
                skdir = os.path.join(root, d)
                if any(fn.lower().endswith(".jpg") for fn in os.listdir(skdir)):
                    skeleton_dirs.append(skdir)

    console_step("Rilevamento cartelle skeleton")
    logging.info(f"Trovate {len(skeleton_dirs)} cartelle 'skeleton' con JPG.")
    print(f"→ {len(skeleton_dirs)} cluster trovati")

    if not skeleton_dirs:
        return

    for skdir in skeleton_dirs:
        process_skeleton_dir(skel_dir=skdir, output_base=output_base, input_base=input_base, params=None, max_workers=max_workers)
        logging.info(f"Cluster completato: {skdir}")

    console_step("Elaborazione completata")
    logging.info("Elaborazione completata per tutti i cluster")
    print(f"{Fore.CYAN}✨ Tutti i cluster processati! ✨{Style.RESET_ALL}")

# ====================================================
# ENTRYPOINT
# ====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minutiae Extraction Runner")
    parser.add_argument("--input", type=str, default="dataset/processed/debug")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    console_step("Avvio Estrazione Minutiae")
    print(f"Input directory : {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Workers         : {args.workers or 'auto'}")

    main(input_base=args.input, output_base=args.output, max_workers=args.workers)
