"""
enhancement.py (robust)
-----------------------
Pipeline di preprocessing fingerprint:
- Normalizzazione robusta
- Smoothing anisotropico (TV) + Bilateral (opzionale)
- Gabor enhancement
- ROI extraction (usata come mask)
- Binarizzazione dentro ROI
- Thinning / skeletonization corretta dentro ROI
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import skeletonize, thin
from typing import Dict

# Try to detect OpenCV thinning
USE_OPENCV_THINNING = False
try:
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        USE_OPENCV_THINNING = True
except Exception:
    USE_OPENCV_THINNING = False


# --------------------------
# Normalizzazione
# --------------------------
def normalize_image(img: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance then scale 0..255 (uint8)."""
    img = img.astype(np.float32)
    mu, sigma = img.mean(), img.std()
    sigma = max(sigma, 1e-6)
    norm = (img - mu) / sigma
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


# --------------------------
# Smoothing
# --------------------------
def anisotropic_smoothing(img: np.ndarray, weight: float = 0.08, iterations: int = 20) -> np.ndarray:
    """TV denoise (Perona-Malik-like). Input 0..255 uint8."""
    img_norm = img.astype(np.float32) / 255.0
    den = denoise_tv_chambolle(img_norm, weight=weight, n_iter_max=iterations)
    return (np.clip(den, 0.0, 1.0) * 255).astype(np.uint8)


def bilateral_smoothing(img: np.ndarray, d: int = 7, sigma_color: float = 30, sigma_space: float = 15) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# --------------------------
# Gabor enhancement
# --------------------------
def gabor_filter_bank(num_orientations: int = 8, ksize: int = 19, sigma: float = 4.0,
                      lambd: float = 8.0, gamma: float = 0.5):
    kernels = []
    for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        s = kern.sum()
        if abs(s) > 1e-6:
            kern /= s
        kernels.append(kern)
    return kernels


def apply_gabor_enhancement(img: np.ndarray, kernels) -> np.ndarray:
    img_f = img.astype(np.float32)
    responses = [cv2.filter2D(img_f, cv2.CV_32F, k) for k in kernels]
    # Combine by magnitude (max) and normalize
    combined = np.max(np.stack(responses, axis=0), axis=0)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    combined_uint8 = combined.astype(np.uint8)
    # Light CLAHE to boost ridge contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(combined_uint8)


# --------------------------
# ROI extraction
# --------------------------
def extract_roi(img: np.ndarray, block_size: int = 16, threshold_rel: float = 0.15) -> np.ndarray:
    """
    ROI: local variance-based mask.
    Works better than global threshold in many fingerprint images.
    Returns binary mask uint8 0/255.
    """
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)

    # compute local std by blocks
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y + block_size, x:x + block_size]
            if block.size == 0:
                continue
            if block.std() > (np.mean(img) * threshold_rel):
                mask[y:y + block_size, x:x + block_size] = 255

    # morphological cleanups
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return mask


# --------------------------
# Binarization & skeleton
# --------------------------
def binarize_image(enhanced: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
    """Adaptive binarization inside ROI if provided."""
    # apply light Gaussian blur to stabilize thresholding
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # adaptive threshold
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 7)

    # refine with Otsu within ROI (if ROI present)
    if roi_mask is not None and roi_mask.sum() > 0:
        masked = cv2.bitwise_and(blur, blur, mask=(roi_mask > 0).astype(np.uint8) * 255)
        # if masked area valid compute otsu on that area else fallback
        try:
            _, otsu_th = cv2.threshold(masked[roi_mask > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_bin = (blur > otsu_th).astype(np.uint8) * 255
            combined = cv2.bitwise_and(adaptive, otsu_bin)
        except Exception:
            combined = adaptive
    else:
        _, otsu_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_bin = (blur > otsu_th).astype(np.uint8) * 255
        combined = cv2.bitwise_and(adaptive, otsu_bin)

    # morphological close to join broken ridges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return combined


def thinning(binary: np.ndarray) -> np.ndarray:
    """Return a 0/255 uint8 skeleton using best available thinning."""
    # Ensure binary is 0/255
    bin_u8 = (binary > 0).astype(np.uint8) * 255

    if USE_OPENCV_THINNING:
        thin_cv = cv2.ximgproc.thinning(bin_u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thin_cv
    else:
        # skimage.thin expects boolean foreground==True
        bool_fore = (bin_u8 == 255)
        # thin gives boolean array
        th = thin(bool_fore)
        return (th.astype(np.uint8) * 255)


# --------------------------
# Pipeline completa
# --------------------------
def preprocess_fingerprint(img: np.ndarray, do_bilateral: bool = True) -> Dict[str, np.ndarray]:
    """
    Steps:
      - normalize
      - anisotropic smoothing
      - gabor enhancement
      - bilateral smoothing (optional)
      - roi extraction (mask)
      - binarize (inside roi)
      - thin/skeletonize inside roi
    """
    # ensure grayscale uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    normalized = normalize_image(img)
    smoothed = anisotropic_smoothing(normalized, weight=0.08, iterations=18)
    kernels = gabor_filter_bank()
    enhanced = apply_gabor_enhancement(smoothed, kernels)
    if do_bilateral:
        enhanced = bilateral_smoothing(enhanced, d=7, sigma_color=25, sigma_space=12)

    roi_mask = extract_roi(enhanced)

    # mask enhanced for binarization stability
    if roi_mask.sum() > 0:
        enhanced_masked = cv2.bitwise_and(enhanced, enhanced, mask=(roi_mask > 0).astype(np.uint8) * 255)
    else:
        enhanced_masked = enhanced.copy()

    binary = binarize_image(enhanced_masked, roi_mask=roi_mask)

    # skeleton only inside ROI to avoid artifacts
    if roi_mask.sum() > 0:
        # zero outside ROI
        binary_roi = cv2.bitwise_and(binary, binary, mask=(roi_mask > 0).astype(np.uint8) * 255)
    else:
        binary_roi = binary

    skeleton = thinning(binary_roi)

    return {
        "normalized": normalized,
        "smoothed": smoothed,             # <- aggiunto
        "enhanced": enhanced,
        "enhanced_masked": enhanced_masked,
        "roi_mask": roi_mask,
        "binary": binary_roi,
        "skeleton": skeleton
    }