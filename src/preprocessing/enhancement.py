import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, label, opening, closing, disk

# ==========================
# NORMALIZZAZIONE + CLAHE
# ==========================
def normalize_image(img: np.ndarray, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

# ==========================
# DENOISING POTENTE
# ==========================
def denoise_image(img: np.ndarray) -> np.ndarray:
    nlm = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    bilateral = cv2.bilateralFilter(nlm, d=3, sigmaColor=30, sigmaSpace=3)
    smooth = cv2.GaussianBlur(bilateral, (3,3), 0.5)
    return smooth

# ==========================
# BINARIZZAZIONE MULTI-STEP
# ==========================
def binarize_image(img: np.ndarray, window_size=15, k=0.15, min_size=15, max_hole=20, patch_size=64) -> np.ndarray:
    thresh = threshold_sauvola(img, window_size=window_size, k=k)
    binary = img < thresh
    
    h, w = binary.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.size == 0: 
                continue
            otsu_thresh = threshold_otsu(patch)
            patch_bin = patch < otsu_thresh
            binary[i:i+patch_size, j:j+patch_size] = np.logical_or(binary[i:i+patch_size, j:j+patch_size], patch_bin)
    
    binary = opening(binary, disk(1))
    binary = closing(binary, disk(1))
    
    binary = remove_small_holes(binary, area_threshold=max_hole)
    
    labeled = label(binary)
    clean = remove_small_objects(labeled, min_size=min_size)
    
    return (clean > 0).astype(np.uint8) * 255

# ==========================
# MASCHERATURA BORDI
# ==========================
def mask_borders(binary_img: np.ndarray, margin=5) -> np.ndarray:
    img = binary_img.copy()
    img[:margin, :] = 0
    img[-margin:, :] = 0
    img[:, :margin] = 0
    img[:, -margin:] = 0
    return img

# ==========================
# SKELETON
# ==========================
def thinning_opencv(binary_img: np.ndarray) -> np.ndarray:
    import cv2.ximgproc as xip
    return xip.thinning(binary_img, thinningType=xip.THINNING_GUOHALL)

def clean_skeleton(skel: np.ndarray, min_size=5) -> np.ndarray:
    skel_bin = (skel > 127).astype(np.uint8)
    neighbors = cv2.filter2D(skel_bin, -1, np.ones((3,3), np.uint8))
    skel_bin[(skel_bin==1) & (neighbors<=1)] = 0
    labeled = label(skel_bin)
    clean = remove_small_objects(labeled, min_size=min_size)
    return (clean > 0).astype(np.uint8) * 255

# ==========================
# PIPELINE COMPLETA
# ==========================
def preprocess_fingerprint(img: np.ndarray, debug_dir: str = None) -> dict:
    try:
        normalized = normalize_image(img)
        denoised = denoise_image(normalized)
        binary = binarize_image(denoised)
        binary = mask_borders(binary, margin=8)
        skeleton = thinning_opencv(binary)
        skeleton = clean_skeleton(skeleton)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "normalized.png"), normalized)
            cv2.imwrite(os.path.join(debug_dir, "denoised.png"), denoised)
            cv2.imwrite(os.path.join(debug_dir, "binary.png"), binary)
            cv2.imwrite(os.path.join(debug_dir, "skeleton.png"), skeleton)

        return {
            "normalized": normalized,
            "denoised": denoised,
            "binary": binary,
            "skeleton": skeleton
        }

    except Exception as e:
        print(f"⚠️ preprocess_fingerprint failed: {e}")
        return {k: None for k in ["normalized","denoised","binary","skeleton"]}
