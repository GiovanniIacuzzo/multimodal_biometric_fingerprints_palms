"""
enhancement.py
--------------
Modulo per il preprocessing delle immagini di impronte digitali:
- Normalizzazione
- Filtraggio Gabor orientato (enhancement)
- Binarizzazione
- Skeletonization
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter

# ==========================
# NORMALIZZAZIONE
# ==========================

def normalize_image(img: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """
    Normalizza l'immagine in modo che abbia la media e deviazione standard desiderate.
    """
    img = img.astype(np.float32)
    img_mean, img_std = img.mean(), img.std()
    normed = (img - img_mean) / (img_std + 1e-8)
    normed = mean + normed * std
    return np.clip(normed, 0, 255).astype(np.uint8)


# ==========================
# FILTRI GABOR (ENHANCEMENT)
# ==========================

def gabor_filter_bank(num_orientations=8, ksize=21, sigma=5, lambd=8, gamma=0.5):
    """
    Crea una banca di filtri Gabor a diverse orientazioni.
    """
    filters = []
    for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def apply_gabor_enhancement(img: np.ndarray, filters) -> np.ndarray:
    """
    Applica il filtraggio Gabor e combina le risposte massime per evidenziare le creste.
    """
    img = img.astype(np.float32)
    responses = [cv2.filter2D(img, cv2.CV_32F, f) for f in filters]
    enhanced = np.max(responses, axis=0)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)


# ==========================
# ROI EXTRACTION
# ==========================

def extract_roi(img: np.ndarray, threshold=15) -> np.ndarray:
    """
    Estrae la regione utile (fingerprint area) usando sogliatura e morfologia.
    """
    blurred = gaussian_filter(img, sigma=3)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    return mask


# ==========================
# BINARIZATION & SKELETONIZATION
# ==========================

def binarize_image(img: np.ndarray) -> np.ndarray:
    """
    Applica una binarizzazione adattiva (Otsu o adattiva).
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def skeletonize_image(binary_img: np.ndarray) -> np.ndarray:
    """
    Riduce le creste a uno scheletro 1-pixel di spessore.
    """
    bin_inv = (binary_img == 0).astype(np.uint8)
    skeleton = skeletonize(bin_inv).astype(np.uint8) * 255
    return skeleton


# ==========================
# PIPELINE COMPLETA
# ==========================

def preprocess_fingerprint(img: np.ndarray) -> dict:
    """
    Applica la pipeline completa:
        normalizzazione → enhancement → ROI → binarizzazione → skeleton.
    Ritorna un dizionario con tutte le immagini intermedie.
    """
    normalized = normalize_image(img)
    filters = gabor_filter_bank()
    enhanced = apply_gabor_enhancement(normalized, filters)
    roi_mask = extract_roi(enhanced)
    binary = binarize_image(enhanced)
    skeleton = skeletonize_image(binary)

    return {
        "normalized": normalized,
        "enhanced": enhanced,
        "roi_mask": roi_mask,
        "binary": binary,
        "skeleton": skeleton
    }
