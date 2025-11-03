"""
enhancement.py (versione robusta 2.4)
-------------------------------------
Pipeline robusta per preprocessing impronte digitali con skeletoning migliorato e logging.
- Normalizzazione con contrast stretching
- Gabor multiorientamento e multiscala
- Smoothing selettivo (gauss + bilaterale)
- ROI adattivo
- Binarizzazione locale (Sauvola)
- Rinforzo creste sottili (closing + dilatazione)
- Skeletonization robusta
- Logging / debug opzionale
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
import os

# ==========================
# NORMALIZZAZIONE
# ==========================
def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_val, max_val = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - min_val) / (max_val - min_val + 1e-8) * 255, 0, 255)
    return img.astype(np.uint8)

# ==========================
# GABOR MULTI-ORIENTATION
# ==========================
def gabor_filter_bank(num_orientations=16, ksize=21, sigma=4.0, lambd=10.0, gamma=0.5):
    filters = []
    for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kern /= np.sum(np.abs(kern)) + 1e-8
        filters.append(kern)
    return filters

def apply_gabor_enhancement(img: np.ndarray, filters) -> np.ndarray:
    responses = [cv2.filter2D(img.astype(np.float32), cv2.CV_32F, f) for f in filters]
    enhanced = np.max(responses, axis=0)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)

# ==========================
# SMOOTHING SELETTIVO
# ==========================
def smooth_image(img: np.ndarray, sigma=0.8, bilateral=True) -> np.ndarray:
    smoothed = gaussian_filter(img, sigma=sigma)
    if bilateral:
        smoothed = cv2.bilateralFilter(smoothed.astype(np.uint8), d=5, sigmaColor=50, sigmaSpace=50)
    return smoothed.astype(np.uint8)

# ==========================
# BINARIZZAZIONE LOCALE
# ==========================
def binarize_image(img: np.ndarray) -> np.ndarray:
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    binary = (img < thresh_sauvola).astype(np.uint8) * 255
    binary = remove_small_objects(binary.astype(bool), min_size=15).astype(np.uint8) * 255
    return binary

# ==========================
# ROI ADATTIVO
# ==========================
def extract_roi(binary_img: np.ndarray, block_size=32, var_thresh=5) -> np.ndarray:
    h, w = binary_img.shape
    mask = np.zeros_like(binary_img, dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = binary_img[y:y+block_size, x:x+block_size]
            if np.var(block) > var_thresh:
                mask[y:y+block_size, x:x+block_size] = 255
    mask = binary_opening(mask>0, structure=np.ones((3,3))).astype(np.uint8) * 255
    mask = binary_closing(mask>0, structure=np.ones((5,5))).astype(np.uint8) * 255
    return mask

# ==========================
# RINFORZO CRESTE PRIMA SKELETON
# ==========================
def reinforce_ridges(binary_img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated

# ==========================
# SKELETONIZATION ROBUSTA
# ==========================
def skeletonize_image(binary_img: np.ndarray, min_branch_size=5) -> np.ndarray:
    reinforced = reinforce_ridges(binary_img)
    crests = (reinforced > 0)
    print(f"[skeletonize] pre_skeleton active_pixels={np.sum(crests)}")
    skeleton = skeletonize(crests).astype(np.uint8)
    skeleton_clean = remove_small_objects(skeleton.astype(bool), min_size=min_branch_size)
    skeleton_final = skeleton_clean.astype(np.uint8) * 255
    print(f"[skeletonize] post_skeleton active_pixels={np.sum(skeleton_final>0)}")
    return skeleton_final

# ==========================
# PIPELINE COMPLETA
# ==========================
def preprocess_fingerprint(img: np.ndarray, debug_dir: str = None) -> dict:
    normalized = normalize_image(img)
    filters = gabor_filter_bank()
    enhanced = apply_gabor_enhancement(normalized, filters)
    smoothed = smooth_image(enhanced, sigma=0.8, bilateral=True)
    binary = binarize_image(smoothed)
    roi_mask = extract_roi(binary)
    binary = cv2.bitwise_and(binary, roi_mask)
    skeleton = skeletonize_image(binary)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "normalized.png"), normalized)
        cv2.imwrite(os.path.join(debug_dir, "enhanced.png"), enhanced)
        cv2.imwrite(os.path.join(debug_dir, "smoothed.png"), smoothed)
        cv2.imwrite(os.path.join(debug_dir, "binary.png"), binary)
        cv2.imwrite(os.path.join(debug_dir, "roi_mask.png"), roi_mask)
        cv2.imwrite(os.path.join(debug_dir, "skeleton.png"), skeleton)

    return {
        "normalized": normalized,
        "enhanced": enhanced,
        "smoothed": smoothed,
        "binary": binary,
        "roi_mask": roi_mask,
        "skeleton": skeleton
    }
