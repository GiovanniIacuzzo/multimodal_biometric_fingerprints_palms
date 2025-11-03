"""
enhancement.py
-------------------------------------
Pipeline avanzata per preprocessing impronte digitali con skeletoning quasi perfetto.
- Normalizzazione con contrast stretching
- Gabor multiorientamento e multiscala
- Smoothing leggero (Gaussian + Bilateral)
- ROI adattivo
- Binarizzazione soft (Sauvola)
- Rinforzo creste direzionale (Top-Hat + Morphology)
- Skeleton su immagine float con thinning conservativo
- Densificazione locale e pruning intelligente
- Logging / debug opzionale
"""

import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import thin
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing

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
# BINARIZZAZIONE SOFT
# ==========================
def binarize_image(img: np.ndarray, k=0.15) -> np.ndarray:
    thresh_sauvola = threshold_sauvola(img, window_size=25, k=k)
    binary = (img < thresh_sauvola).astype(np.uint8) * 255
    binary = binary_opening(binary > 0, structure=np.ones((2, 2))).astype(np.uint8) * 255
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
    mask = binary_opening(mask > 0, structure=np.ones((3, 3))).astype(np.uint8) * 255
    mask = binary_closing(mask > 0, structure=np.ones((5, 5))).astype(np.uint8) * 255
    return mask

# ==========================
# RINFORZO CRESTE DIREZIONALE
# ==========================
def reinforce_ridges_directional(img: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    kernel2 = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, kernel2, iterations=1)
    return closed

# ==========================
# SKELETON CONSERVATIVO + DENSIFICAZIONE
# ==========================
def skeletonize_image(img: np.ndarray, min_branch_size=2) -> np.ndarray:
    arr = np.array(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    img_f = (arr / 255.0).astype(np.float32)
    skeleton_bool = thin(img_f > 0.01).astype(np.uint8)
    skeleton = (skeleton_bool.astype(np.uint8) * 255)

    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)
    return skeleton.astype(np.uint8)

# ==========================
# PIPELINE COMPLETA AVANZATA
# ==========================
def preprocess_fingerprint(img: np.ndarray, debug_dir: str = None) -> dict:
    normalized = normalize_image(img)
    filters = gabor_filter_bank()
    enhanced = apply_gabor_enhancement(normalized, filters)
    smoothed = smooth_image(enhanced, sigma=0.7, bilateral=True)
    binary = binarize_image(smoothed, k=0.15)
    roi_mask = extract_roi(binary)
    binary_roi = cv2.bitwise_and(binary, roi_mask)
    reinforced = reinforce_ridges_directional(binary_roi)
    skeleton = skeletonize_image(reinforced)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "normalized.png"), normalized)
        cv2.imwrite(os.path.join(debug_dir, "enhanced.png"), enhanced)
        cv2.imwrite(os.path.join(debug_dir, "smoothed.png"), smoothed)
        cv2.imwrite(os.path.join(debug_dir, "binary.png"), binary)
        cv2.imwrite(os.path.join(debug_dir, "roi_mask.png"), roi_mask)
        cv2.imwrite(os.path.join(debug_dir, "binary_roi.png"), binary_roi)
        cv2.imwrite(os.path.join(debug_dir, "reinforced.png"), reinforced)
        cv2.imwrite(os.path.join(debug_dir, "skeleton.png"), skeleton)

    return {
        "normalized": normalized,
        "enhanced": enhanced,
        "smoothed": smoothed,
        "binary": binary,
        "roi_mask": roi_mask,
        "binary_roi": binary_roi,
        "reinforced": reinforced,
        "skeleton": skeleton
    }
