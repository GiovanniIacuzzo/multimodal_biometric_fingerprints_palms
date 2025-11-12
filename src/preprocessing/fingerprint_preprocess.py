import os
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, skeletonize
from scipy.ndimage import gaussian_filter, convolve, sobel
from src.preprocessing.orientation import compute_orientation_map, visualize_orientation
from config import config_fingerprint

# ================================================
# NORMALIZATION + CLAHE
# ================================================
def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalizza immagine e applica CLAHE in modo robusto."""
    if img.dtype == np.uint8:
        f = img.astype(np.float32) / 255.0
    else:
        f = (img - img.min()) / (img.max() - img.min() + 1e-8)

    f = (f - np.percentile(f, 1)) / (np.percentile(f, 99) - np.percentile(f, 1) + 1e-12)
    f = np.clip(f, 0.0, 1.0)
    img_u8 = (f * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    return clahe.apply(img_u8)

# ================================================
# DENOISING
# ================================================
def denoise_image(img: np.ndarray) -> np.ndarray:
    """Riduzione rumore bilaterale + gaussian blur per preservare creste."""
    bilateral = cv2.bilateralFilter(img, d=7, sigmaColor=60, sigmaSpace=60)
    return cv2.GaussianBlur(bilateral, (3, 3), 0.6)

# ================================================
# BINARIZATION (Sauvola + Otsu refinements)
# ================================================
def binarize(img: np.ndarray) -> np.ndarray:
    """Binarizzazione ibrida con Sauvola e Otsu locali."""
    img_f = img.astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_f.astype(np.uint8)).astype(np.float32)

    win = 25
    k = 0.25
    mean = cv2.boxFilter(img_eq, -1, (win, win))
    sqmean = cv2.boxFilter(img_eq**2, -1, (win, win))
    std = np.sqrt(np.clip(sqmean - mean**2, 0, None))
    std_n = std / (std.max() + 1e-6)
    k_map = k * (1 - 0.5 * std_n)
    sauv_map = mean * (1 - k_map * (1 - std / (mean + 1e-6)))
    binary = img_eq < sauv_map

    # Refinement Otsu su patch
    patch = 48
    h, w = img_eq.shape
    for i in range(0, h, patch):
        for j in range(0, w, patch):
            sub = img_eq[i:i+patch, j:j+patch]
            if sub.size < 10 or sub.std() < 3:
                continue
            try:
                t = threshold_otsu(sub)
                binary[i:i+patch, j:j+patch] |= (sub < t)
            except Exception:
                pass

    cleaned = remove_small_objects(binary, min_size=80)
    cleaned = remove_small_holes(cleaned, area_threshold=150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    marker = cv2.erode(opened, kernel, iterations=1).astype(bool)
    recon = reconstruction(marker, opened, method='dilation')
    return (recon > 0).astype(np.uint8) * 255

# ================================================
# SEGMENTATION
# ================================================
def segment_fingerprint(img: np.ndarray, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Segmentazione robusta impronta + maschera foreground."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    stab = clahe.apply(gray)
    blur = cv2.GaussianBlur(stab, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, np.ones_like(gray, dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    mask_hull = np.zeros_like(mask)
    cv2.drawContours(mask_hull, [hull], -1, 255, -1)

    x, y, w_box, h_box = cv2.boundingRect(hull)
    margin = 10
    cropped = gray[max(0, y-margin):y+h_box+margin, max(0, x-margin):x+w_box+margin]
    cropped_mask = mask_hull[max(0, y-margin):y+h_box+margin, max(0, x-margin):x+w_box+margin]

    cropped = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)
    return cropped, cropped_mask

# ================================================
# SMOOTHING + THINNING
# ================================================
def smooth_fingerprint_skeleton(binary_img: np.ndarray,
                                sigma: float = 1.4,
                                diffusion_iter: int = 3,
                                contrast_boost: float = 1.25) -> np.ndarray:
    """Smussamento anisotropico per scheletro pulito e continuo."""
    img = binary_img.astype(np.float32) / 255.0
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx, ny = gx / mag, gy / mag

    smoothed = img.copy()
    for _ in range(diffusion_iter):
        dx, dy = sobel(smoothed, axis=1), sobel(smoothed, axis=0)
        grad_proj = dx * ny - dy * nx
        smoothed += sigma * grad_proj

    smoothed = gaussian_filter(smoothed, sigma=0.6)
    smoothed = np.clip(smoothed * contrast_boost, 0, 1)
    return (smoothed > 0.35).astype(np.uint8) * 255

def thinning_and_cleaning(binary_img: np.ndarray,
                          orientation_img: np.ndarray,
                          reliability_img: np.ndarray,
                          rel_thresh: float = 0.1) -> np.ndarray:
    """Scheletizzazione guidata da affidabilità."""
    mask = (binary_img > 0).astype(bool)
    mask = remove_small_objects(mask, min_size=64)
    mask = remove_small_holes(mask, area_threshold=80)
    rel_smooth = gaussian_filter(reliability_img, sigma=2.0)
    mask = mask & (rel_smooth > rel_thresh)
    skeleton = skeletonize(mask)

    # Rimuove terminazioni isolate
    kernel = np.ones((3,3), np.uint8)
    neighbor_count = convolve(skeleton.astype(np.uint8), kernel)
    skeleton = skeleton & (neighbor_count > 1)
    return (skeleton > 0).astype(np.uint8) * 255

# ================================================
# MAIN PIPELINE
# ================================================
def preprocess_fingerprint(img: np.ndarray, debug_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    try:
        normalized = normalize_image(img)
        denoised = denoise_image(normalized)
        segmented, mask = segment_fingerprint(denoised, debug_dir)
        binary = binarize(segmented)

        orient_blocks, orient_img, reliability = compute_orientation_map(
            segmented, block_size=16, smooth_sigma=3.0, invert_if_needed=True,
            smooth_orientation_sigma=3.0, mask=mask
        )

        binary_smooth = smooth_fingerprint_skeleton(binary)
        skeleton = thinning_and_cleaning(binary_smooth, orient_img, reliability)

        orientation_vis = visualize_orientation(
            img=segmented, orient_img=orient_img, reliability_img=reliability,
            block_size=16, scale=7, rel_thresh=0.1, mask=mask
        )

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "normalized.jpg"), normalized)
            cv2.imwrite(os.path.join(debug_dir, "denoised.jpg"), denoised)
            cv2.imwrite(os.path.join(debug_dir, "segmented.jpg"), segmented)
            cv2.imwrite(os.path.join(debug_dir, "binary.jpg"), binary)
            cv2.imwrite(os.path.join(debug_dir, "skeleton.jpg"), skeleton)
            cv2.imwrite(os.path.join(debug_dir, "orientation_vis.jpg"), orientation_vis)

        return {
            "normalized": normalized,
            "denoised": denoised,
            "segmented": segmented,
            "mask": mask,
            "binary": binary,
            "skeleton": skeleton,
            "orientation_blocks": orient_blocks,
            "orientation_map": orient_img,
            "orientation_vis": orientation_vis,
            "reliability": reliability,
        }

    except Exception as e:
        raise RuntimeError(f"preprocess_fingerprint failed: {e}") from e

# ================================================
# TEST LOCALE
# ================================================
if __name__ == "__main__":
    test_path = "135_1_3_skeleton.jpg"  # cambia se serve
    debug_out = "debug_output"
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {test_path}")
    results = preprocess_fingerprint(img, debug_dir=debug_out)
    print(f"Preprocessing completato. Risultati salvati in '{debug_out}'")
