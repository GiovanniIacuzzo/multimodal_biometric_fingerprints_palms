import os
import json
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.features.post_processing import postprocess_minutiae
from src.db.database import get_image_id_by_filename, save_minutiae, save_features_summary

# ============================================================
# CONFIGURAZIONE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================
def thin_skeleton(img: np.ndarray) -> np.ndarray:
    img_u8 = (img > 0).astype(np.uint8) * 255 if img.dtype != np.uint8 else img.copy()
    try:
        thin = cv2.ximgproc.thinning(img_u8)
        return (thin > 0).astype(np.uint8)
    except Exception:
        sk = img_u8.copy()
        prev = np.zeros_like(sk)
        for _ in range(100):
            eroded = cv2.erode(sk, np.ones((3, 3), np.uint8))
            opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
            sk = eroded.copy()
            if np.array_equal(sk, prev):
                break
            prev = sk.copy()
        return (sk > 0).astype(np.uint8)


def extract_minutiae_from_skeleton(skel: np.ndarray) -> List[Dict]:
    sk_gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY) if skel.ndim == 3 else skel.copy()
    bin_pos = (sk_gray > 127).astype(np.uint8)
    bin_neg = (sk_gray <= 127).astype(np.uint8)
    sk = bin_pos if bin_pos.sum() <= bin_neg.sum() else bin_neg

    sk_thin = thin_skeleton((sk * 255).astype(np.uint8))
    if sk_thin.sum() == 0:
        sk_thin = sk

    h, w = sk_thin.shape
    minutiae: List[Dict] = []
    ys, xs = np.nonzero(sk_thin)
    for y, x in zip(ys, xs):
        if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
            continue
        P = [
            sk_thin[y, x + 1], sk_thin[y - 1, x + 1],
            sk_thin[y - 1, x], sk_thin[y - 1, x - 1],
            sk_thin[y, x - 1], sk_thin[y + 1, x - 1],
            sk_thin[y + 1, x], sk_thin[y + 1, x + 1],
        ]
        CN = sum(abs(int(P[i]) - int(P[(i + 1) % 8])) for i in range(8)) // 2
        if CN == 1:
            minutiae.append({"x": int(x), "y": int(y), "type": "ending"})
        elif CN == 3:
            minutiae.append({"x": int(x), "y": int(y), "type": "bifurcation"})
    return minutiae


# ============================================================
# ELABORAZIONE SINGOLO SAMPLE
# ============================================================
def process_sample(debug_dir: str, output_dir: str, params: Optional[Dict] = None) -> None:
    sample_name = os.path.basename(debug_dir.rstrip("/\\"))
    skel_path = os.path.join(debug_dir, f"{sample_name}_skeleton.jpg")
    gray_path = os.path.join(debug_dir, f"{sample_name}_segmented.jpg")

    if not os.path.exists(skel_path):
        logging.error(f"File skeleton non trovato: {skel_path}")
        return

    skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gray_path) else skel
    if skel is None or gray is None:
        logging.error(f"Errore caricamento immagini per '{sample_name}'.")
        return

    start_time = time.time()
    raw_minutiae = extract_minutiae_from_skeleton(skel)
    refined = postprocess_minutiae(raw_minutiae, skel, gray, params)
    duration = time.time() - start_time

    image_id = get_image_id_by_filename(f"{sample_name}.jpg")
    if image_id is None:
        logging.warning(f"Nessuna voce trovata nel database per '{sample_name}.jpg'.")
        return

    if refined:
        save_minutiae(image_id, refined)

    avg_quality = float(np.mean([m.get("quality", 0) for m in refined])) if refined else 0.0
    avg_coherence = float(np.mean([m.get("coherence", 0) for m in refined])) if refined else 0.0

    save_features_summary(
        image_id=image_id,
        raw_count=len(raw_minutiae),
        post_count=len(refined),
        avg_quality=avg_quality,
        avg_coherence=avg_coherence,
        processing_time_sec=duration,
        params=params or {}
    )

    os.makedirs(output_dir, exist_ok=True)
    img_out = os.path.join(output_dir, f"{sample_name}_minutiae_postprocessed.jpg")
    json_out = os.path.join(output_dir, f"{sample_name}_minutiae.json")

    vis = cv2.cvtColor((skel > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for m in refined:
        color = (0, 0, 255) if m.get("type") == "ending" else (0, 255, 0)
        cv2.circle(vis, (int(m["x"]), int(m["y"])), 3, color, -1)

    cv2.imwrite(img_out, vis)
    with open(json_out, "w") as f:
        json.dump(refined, f, indent=2)


# ============================================================
# MAIN – ELABORAZIONE DI TUTTI I CLUSTER
# ============================================================
def main(input_base: Optional[str] = None, output_base: Optional[str] = None, max_workers: int = None):
    input_base = input_base or os.path.join("data", "processed", "debug")
    output_base = output_base or os.path.join("data", "features", "minutiae")

    if not os.path.exists(input_base):
        raise FileNotFoundError(f"Cartella non trovata: {input_base}")
    os.makedirs(output_base, exist_ok=True)

    # CERCA TUTTI I SAMPLE RICORSIVAMENTE
    sample_dirs = []
    for root, dirs, files in os.walk(input_base):
        for d in dirs:
            path = os.path.join(root, d)
            # Riconosciamo una "cartella di sample" se contiene skeleton o segmented
            if any(fname.endswith("_skeleton.jpg") for fname in os.listdir(path)):
                sample_dirs.append(path)

    if not sample_dirs:
        logging.warning("Nessuna cartella contenente impronte trovata.")
        return

    logging.info(f"Trovate {len(sample_dirs)} impronte da elaborare in cluster multipli.\n")

    # Esecuzione parallela mantenendo struttura cluster
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for debug_dir in sample_dirs:
            # Mantieni struttura cluster per output
            rel_path = os.path.relpath(debug_dir, input_base)
            out_dir = os.path.join(output_base, rel_path)
            futures[executor.submit(process_sample, debug_dir, out_dir)] = debug_dir

        for future in tqdm(as_completed(futures), total=len(futures), desc="Elaborazione impronte", unit="impronta"):
            debug_dir = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Errore durante l'elaborazione di '{debug_dir}': {e}")

    logging.info("\nElaborazione batch completata.")
