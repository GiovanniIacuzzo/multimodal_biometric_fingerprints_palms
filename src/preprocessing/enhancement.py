import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import thin, remove_small_objects, label
from scipy.ndimage import binary_opening, binary_closing

# ==========================
# NORMALIZZAZIONE + CLAHE
# ==========================
def normalize_image(img: np.ndarray, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    """
    Normalizza l'immagine e applica CLAHE per contrasto locale.
    """
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    norm = clahe.apply(img)
    return norm

# ==========================
# DENOISING LEGERO
# ==========================
def denoise_image(img: np.ndarray, method="median", ksize=3) -> np.ndarray:
    """
    Rimuove piccoli artefatti senza sfumare le creste.
    """
    if method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0.5)
    else:
        return img

# ==========================
# BINARIZZAZIONE LOCALE
# ==========================
def binarize_image(img: np.ndarray, window_size=25, k=0.1) -> np.ndarray:
    thresh = threshold_sauvola(img, window_size=window_size, k=k)
    binary = img < thresh  # True sulle creste
    # Apertura/chiusura delicata per pulizia
    binary = binary_opening(binary, structure=np.ones((1,1)))
    binary = binary_closing(binary, structure=np.ones((1,1)))
    return (binary.astype(np.uint8) * 255)  # OpenCV vuole 0/255

# ==========================
# THINNING ROBUSTO
# ==========================
def thinning_opencv(binary_img: np.ndarray) -> np.ndarray:
    """
    Skeletonizzazione robusta con OpenCV ximgproc.thinning
    """
    import cv2.ximgproc as xip
    skeleton = xip.thinning(binary_img, thinningType=xip.THINNING_GUOHALL)
    return skeleton

def clean_skeleton(skeleton: np.ndarray, min_component_size=5) -> np.ndarray:
    skel_bin = (skeleton > 127).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    neighbors = cv2.filter2D(skel_bin, -1, kernel)
    isolated = (skel_bin == 1) & (neighbors <= 1)
    skel_bin[isolated] = 0

    labeled = label(skel_bin)
    skel_clean = remove_small_objects(labeled, min_size=min_component_size)
    skel_clean = (skel_clean > 0).astype(np.uint8)

    skel_thin = thin(skel_clean).astype(np.uint8)

    return skel_thin * 255


# ==========================
# PIPELINE COMPLETA OTTIMIZZATA
# ==========================
def preprocess_fingerprint(img: np.ndarray, debug_dir: str = None) -> dict:
    """
    Pipeline completa ottimizzata per fingerprint preprocessing.
    """
    try:
        # --- Normalizzazione e contrasto ---
        normalized = normalize_image(img)

        # --- Denoising leggero ---
        denoised = denoise_image(normalized, method="median", ksize=3)

        # --- Binarizzazione ---
        binary = binarize_image(denoised, window_size=25, k=0.1)

        # --- Thinning / Skeleton ---
        skeleton = thinning_opencv(binary)
        skeleton = clean_skeleton(skeleton)

        # --- Debug: salva tutte le fasi ---
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
            "skeleton": skeleton,
        }

    except Exception as e:
        print(f"⚠️ preprocess_fingerprint failed: {e}")
        return {k: None for k in ["normalized","denoised","binary","skeleton"]}
