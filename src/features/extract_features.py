import os
import cv2
import time
import json
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from scipy.spatial import cKDTree
from src.features.post_processing import postprocess_minutiae
from src.features.utils import (
    ensure_dir, save_debug_image,
    clean_skeleton, compute_neighbor_count
)

logger = logging.getLogger("minutiae_extractor")

# =====================================================
# 1. CN Lookup Table
# =====================================================
_CN_LUT = None
def _build_cn_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for pat in range(256):
        b = [(pat >> i) & 1 for i in range(8)]
        cn = sum(abs(b[i] - b[(i + 1) % 8]) for i in range(8)) // 2
        lut[pat] = cn
    return lut


def compute_cn_map(skel: np.ndarray) -> np.ndarray:
    global _CN_LUT
    if _CN_LUT is None:
        _CN_LUT = _build_cn_lut()
    sk = (skel > 0).astype(np.uint8)
    # 8 neighbors encoding clockwise starting top-left
    kernel = np.array([[1, 2, 4],
                       [128, 0, 8],
                       [64, 32, 16]], dtype=np.uint8)
    pattern = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return _CN_LUT[pattern]


# =====================================================
# 2. Raw minutiae extraction + neighborhood check
# =====================================================
def extract_minutiae_raw(skel: np.ndarray, min_dist: int = 5) -> List[Dict]:
    """Extract minutiae candidates from binary skeleton."""
    if skel is None or skel.size == 0:
        return []
    sk = (skel > 0).astype(np.uint8)
    cn_map = compute_cn_map(sk)
    deg_map = compute_neighbor_count(sk)

    mask = (sk == 1) & ((cn_map == 1) | (cn_map == 3))
    ys, xs = np.nonzero(mask)

    minutiae = []
    for y, x in zip(ys, xs):
        cn, deg = int(cn_map[y, x]), int(deg_map[y, x])
        if cn == 1:
            t = "ending"
        elif cn == 3:
            t = "bifurcation"
        else:
            continue
        minutiae.append({"x": float(x), "y": float(y), "type": t, "cn": cn, "deg": deg})

    # ---- Remove close duplicates
    if len(minutiae) > 1:
        pts = np.array([[m["x"], m["y"]] for m in minutiae])
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=min_dist)
        to_remove = set()
        for i, j in pairs:
            to_remove.add(j)
        minutiae = [m for k, m in enumerate(minutiae) if k not in to_remove]

    return minutiae


# =====================================================
# 3. Local orientation around minutia
# =====================================================
def local_orientation(gray: np.ndarray, x: int, y: int, window: int = 11) -> float:
    h, w = gray.shape
    x1, x2 = max(0, x - window//2), min(w, x + window//2)
    y1, y2 = max(0, y - window//2), min(h, y + window//2)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    ang = 0.5 * np.arctan2(2 * np.mean(gx * gy), np.mean(gx**2 - gy**2) + 1e-8)
    return float(ang)


# =====================================================
# 4. Process skeleton
# =====================================================
def process_skeleton(path: str, params: dict, out_dir: str):
    img_name = os.path.splitext(os.path.basename(path))[0].replace("_skeleton", "")
    subject_id = img_name.split("_")[0]
    subj_dir = os.path.join(out_dir, subject_id)
    ensure_dir(subj_dir)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Immagine non trovata o corrotta: {path}")
        return

    # Pulizia e binarizzazione
    sk = clean_skeleton(img, invert_auto=True)
    sk_bin = (sk > 0).astype(np.uint8)

    raw = extract_minutiae_raw(sk_bin, min_dist=5)

    norm_path = path.replace("_skeleton.png", "_normalized.png")
    gray = cv2.imread(norm_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        gray = sk_bin * 255

    # -------------------------------------------------
    # Post-processing e raffinamento minutiae
    # -------------------------------------------------
    processed = postprocess_minutiae(raw, sk_bin, gray, params)

    json.dump(raw, open(os.path.join(subj_dir, f"{img_name}_raw_minutiae.json"), "w"), indent=2)
    desc_raw = np.array([[m["x"], m["y"],
                          0 if m["type"] == "ending" else 1,
                          m.get("orientation", 0.0)]
                         for m in raw], dtype=np.float32)
    np.save(os.path.join(subj_dir, f"{img_name}_raw_descriptor.npy"), desc_raw)

    # POST-PROCESSED
    json.dump(processed, open(os.path.join(subj_dir, f"{img_name}_minutiae.json"), "w"), indent=2)
    desc_proc = np.array([[m["x"], m["y"],
                           0 if m["type"] == "ending" else 1,
                           m.get("orientation", 0.0),
                           m.get("quality", 0.0),
                           m.get("coherence", 0.0)]
                          for m in processed], dtype=np.float32)
    np.save(os.path.join(subj_dir, f"{img_name}_descriptor.npy"), desc_proc)

    # -------------------------------------------------
    # Visualizzazione RAW e POST
    # -------------------------------------------------
    vis_raw = cv2.cvtColor(sk_bin * 255, cv2.COLOR_GRAY2BGR)
    for m in raw:
        x, y = int(m["x"]), int(m["y"])
        color = (0, 255, 0) if m["type"] == "bifurcation" else (0, 0, 255)
        cv2.circle(vis_raw, (x, y), 4, color, -1)
    save_debug_image(os.path.join(subj_dir, f"{img_name}_raw_minutiae_vis.png"), vis_raw, normalize=False)

    vis_post = cv2.cvtColor(sk_bin * 255, cv2.COLOR_GRAY2BGR)
    for m in processed:
        x, y = int(m["x"]), int(m["y"])
        color = (0, 255, 0) if m["type"] == "bifurcation" else (0, 0, 255)
        cv2.circle(vis_post, (x, y), 4, color, -1)
        # Disegna la direzione
        ang = m.get("orientation", 0.0)
        x2 = int(x + 10 * np.cos(ang))
        y2 = int(y + 10 * np.sin(ang))
        cv2.line(vis_post, (x, y), (x2, y2), (255, 255, 0), 1)
    save_debug_image(os.path.join(subj_dir, f"{img_name}_minutiae_vis.png"), vis_post, normalize=False)

    # -------------------------------------------------
    # Logging finale
    # -------------------------------------------------
    logger.info(f"[{subject_id}] RAW: {len(raw)} | POST: {len(processed)} minutiae salvate in {subj_dir}")


# =====================================================
# 5. Batch entrypoint
# =====================================================
def main(processed_dir="data/processed", features_dir="data/features", params=None,
         debug=False, small_subset=False):
    """
    Esegue la fase di estrazione e post-processing delle minutiae
    per tutte le impronte preprocessate.
    """
    # === 1. Imposta directory di input/output ===
    input_dir = os.path.join(processed_dir, "debug")
    out_dir = os.path.join(features_dir, "minutiae")
    ensure_dir(out_dir)

    subdirs = [d for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]
    paths = [os.path.join(input_dir, d, f"{d}_skeleton.png") for d in subdirs]
    paths = [p for p in paths if os.path.exists(p)]

    if small_subset:
        paths = paths[:10]
        logger.info("Esecuzione in modalità 'small_subset' (solo 10 campioni).")

    logger.info(f"Trovati {len(paths)} skeleton da elaborare.")

    # === 2. Parametri di default (se non forniti) ===
    default_params = {
        "min_distance": 8.0,
        "orientation_window": 11,
        "quality_window": 25,
        "quality_threshold": 0.05,
        "coherence_threshold": 0.05
    }
    if params is None:
        params = default_params
    else:
        default_params.update(params)
        params = default_params

    # === 3. Elaborazione ===
    summary = []
    start_global = time.time()

    for p in tqdm(paths, desc="Estrazione e post-processing minutiae"):
        try:
            start = time.time()
            process_skeleton(p, params, out_dir)
            duration = time.time() - start
            summary.append({"path": p, "time_sec": round(duration, 3)})
        except Exception as e:
            logger.error(f"Errore durante l'elaborazione di {p}: {e}", exc_info=True)

    # === 4. Riassunto e statistiche globali ===
    total_time = time.time() - start_global
    avg_time = total_time / max(1, len(summary))
    logger.info(f"Pipeline completata in {total_time:.2f}s (media {avg_time:.2f}s per immagine).")

    # === 5. Salva report riassuntivo ===
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_images": len(paths),
            "processed": len(summary),
            "average_time_sec": round(avg_time, 3),
            "total_time_sec": round(total_time, 2),
            "params": params,
            "details": summary
        }, f, indent=2)

    logger.info(f"Report salvato in {summary_path}")