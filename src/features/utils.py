import os
import cv2
import numpy as np
from skimage.morphology import thin
import logging

logger = logging.getLogger("minutiae_extractor")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_debug_image(path: str, img: np.ndarray, normalize=True):
    ensure_dir(os.path.dirname(path))
    if normalize:
        img = np.clip((img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + 1e-6), 0, 1)
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def maybe_invert(img: np.ndarray) -> np.ndarray:
    return 255 - img if np.mean(img) > 127 else img

def compute_neighbor_count(skel: np.ndarray) -> np.ndarray:
    sk = (skel > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neigh_no_center = neigh - sk
    return neigh_no_center.astype(np.uint8)

def clean_skeleton(img: np.ndarray, invert_auto=True, prune_iters=0) -> np.ndarray:
    if invert_auto:
        img = maybe_invert(img)
    sk = (img > 0).astype(np.uint8)
    sk = thin(sk).astype(np.uint8)

    deg = compute_neighbor_count(sk)
    junction_mask = (deg >= 3)

    for _ in range(max(0, int(prune_iters))):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
        for lab in range(1, num):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area <= 2:
                comp_mask = (labels == lab)
                if not np.any(junction_mask & comp_mask):
                    sk[comp_mask] = 0
    return (sk > 0).astype(np.uint8) * 255

def compute_orientation_map(gray: np.ndarray, sigma=3.0):
    img = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    gxx, gyy, gxy = gx * gx, gy * gy, gx * gy
    Sxx = cv2.GaussianBlur(gxx, (0, 0), sigma)
    Syy = cv2.GaussianBlur(gyy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(gxy, (0, 0), sigma)
    orient = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    lambda1 = 0.5 * (Sxx + Syy + np.sqrt((Sxx - Syy)**2 + 4 * Sxy**2))
    lambda2 = 0.5 * (Sxx + Syy - np.sqrt((Sxx - Syy)**2 + 4 * Sxy**2))
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    return orient, np.clip(coherence, 0, 1)
