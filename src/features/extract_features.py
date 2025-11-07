# src/features/minutiae_extraction.py
import os
import cv2
import json
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from scipy.spatial import cKDTree

from src.features.utils import (
    ensure_dir, save_debug_image,
    clean_skeleton, compute_neighbor_count, compute_orientation_map
)

logger = logging.getLogger("minutiae_extractor")

# =====================================================
# 1. CN Lookup Table (8-neighborhood clockwise)
# =====================================================
_CN_LUT = None
def _build_cn_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for pat in range(256):
        b = [(pat >> i) & 1 for i in range(8)]  # [p0..p7]
        # CN definition: half of number of transitions between successive pairs in a circle
        cn = sum(abs(b[i] - b[(i + 1) % 8]) for i in range(8)) // 2
        lut[pat] = cn
    return lut


def compute_cn_map(skel: np.ndarray) -> np.ndarray:
    """Compute crossing number map for skeleton."""
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
# 4. Process skeleton (no post-processing yet)
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

    sk = clean_skeleton(img, invert_auto=True)
    sk_bin = (sk > 0).astype(np.uint8)

    raw = extract_minutiae_raw(sk_bin, min_dist=5)

    # compute local orientation for visualization
    gray = cv2.imread(path.replace("_skeleton.png", "_normalized.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        gray = sk_bin * 255

    for m in raw:
        m["orientation"] = local_orientation(gray, int(m["x"]), int(m["y"]))
        m["quality"] = 1.0  # placeholder
        m["coherence"] = 1.0  # placeholder

    # Save JSON + descriptor
    json.dump(raw, open(os.path.join(subj_dir, f"{img_name}_raw_minutiae.json"), "w"), indent=2)
    desc = np.array([[m["x"], m["y"],
                      0 if m["type"] == "ending" else 1,
                      m["orientation"]]
                     for m in raw], dtype=np.float32)
    np.save(os.path.join(subj_dir, f"{img_name}_raw_descriptor.npy"), desc)

    # Visualization
    vis = cv2.cvtColor(sk_bin * 255, cv2.COLOR_GRAY2BGR)
    for m in raw:
        x, y = int(m["x"]), int(m["y"])
        color = (0, 255, 0) if m["type"] == "bifurcation" else (0, 0, 255)
        cv2.circle(vis, (x, y), 4, color, -1)
        x2 = int(x + 10 * np.cos(m["orientation"]))
        y2 = int(y + 10 * np.sin(m["orientation"]))
        cv2.line(vis, (x, y), (x2, y2), (255, 255, 0), 1)

    save_debug_image(os.path.join(subj_dir, f"{img_name}_raw_minutiae_vis.png"), vis, normalize=False)

    logger.info(f"[{subject_id}] raw minutiae: {len(raw)} saved in {subj_dir}")


# =====================================================
# 5. Batch entrypoint
# =====================================================
def main(processed_dir="data/processed", features_dir="data/features", params=None):
    input_dir = os.path.join(processed_dir, "debug")
    out_dir = os.path.join(features_dir, "minutiae")
    ensure_dir(out_dir)
    subdirs = [d for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]
    paths = [os.path.join(input_dir, d, f"{d}_skeleton.png") for d in subdirs]
    paths = [p for p in paths if os.path.exists(p)]

    logger.info(f"Trovati {len(paths)} skeleton da elaborare.")
    for p in tqdm(paths, desc="Estrazione raw minutiae"):
        try:
            process_skeleton(p, params or {}, out_dir)
        except Exception as e:
            logger.error(f"Errore su {p}: {e}", exc_info=True)
