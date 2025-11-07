import os
from typing import Optional
import numpy as np
import cv2
import math
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, skeletonize
from config import config


# ================================================
# UTILITY: salvataggio immagini di debug
# ================================================
def _debug_write(debug_dir: Optional[str], name: str, img: np.ndarray):
    """Scrive immagine di debug in cartella, se abilitato."""
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, name)
        if img.dtype != np.uint8:
            img = (img.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"[DEBUG] Impossibile salvare {name}: {e}")


# ================================================
# NORMALIZATION + CLAHE
# ================================================
def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalizza immagine con CLAHE (contrasto locale adattivo)."""
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=(config.CLAHE_TILE_SIZE, config.CLAHE_TILE_SIZE)
    )
    img_u8 = img.astype(np.uint8)
    return clahe.apply(img_u8)


# ================================================
# DENOISING
# ================================================
def denoise_image(img: np.ndarray) -> np.ndarray:
    """Riduzione rumore con filtro bilaterale + Gaussiano."""
    b = cv2.bilateralFilter(
        img, d=config.BILATERAL_D, sigmaColor=config.BILATERAL_SIGMA_COLOR, sigmaSpace=config.BILATERAL_SIGMA_SPACE
    )
    return cv2.GaussianBlur(b, (3, 3), config.GAUSSIAN_SIGMA)


# ================================================
# BINARIZATION
# ================================================
def robust_binarize(img: np.ndarray) -> np.ndarray:
    """Binarizzazione robusta combinando Sauvola + Otsu locale."""
    img_f = img.astype(np.uint8)
    thresh = threshold_sauvola(img_f, window_size=config.SAUVOLA_WIN, k=config.SAUVOLA_K)
    binary = img_f < thresh

    h, w = img_f.shape
    for i in range(0, h, config.LOCAL_PATCH):
        for j in range(0, w, config.LOCAL_PATCH):
            patch = img_f[i:i + config.LOCAL_PATCH, j:j + config.LOCAL_PATCH]
            if patch.size == 0:
                continue
            try:
                o = threshold_otsu(patch)
                binary[i:i + config.LOCAL_PATCH, j:j + config.LOCAL_PATCH] |= (patch < o)
            except Exception:
                pass

    marker = binary.copy()
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marker = cv2.erode(marker.astype(np.uint8), se, iterations=1).astype(bool)
    mask = binary.astype(bool)
    recon = reconstruction(marker.astype(np.uint8), mask.astype(np.uint8), method='dilation')
    recon = remove_small_objects(recon.astype(bool), min_size=config.MIN_OBJ_SIZE)
    recon = remove_small_holes(recon.astype(bool), area_threshold=config.MAX_HOLE_SIZE)

    return (recon > 0).astype(np.uint8) * 255


# ================================================
# SEGMENTATION
# ================================================
def segment_fingerprint(img: np.ndarray) -> np.ndarray:
    """Ritaglia l'impronta eliminando lo sfondo."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < config.MIN_SEGMENT_AREA:
        return gray

    x, y, w, h = cv2.boundingRect(largest_contour)
    return gray[y:y + h, x:x + w]


# ================================================
# ORIENTATION MAP
# ================================================
def compute_orientation_map(img: np.ndarray):
    """Calcola mappa di orientamento e affidabilità per le ridge."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    h, w = gray.shape
    gray_f = gray.astype(np.float32) / 255.0

    Gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    Vx = 2.0 * (Gx * Gy)
    Vy = (Gx**2 - Gy**2)
    energy = (Gx**2 + Gy**2)

    Vx_s = gaussian_filter(Vx, sigma=config.ORIENT_SIGMA)
    Vy_s = gaussian_filter(Vy, sigma=config.ORIENT_SIGMA)
    energy_s = gaussian_filter(energy, sigma=config.ORIENT_SIGMA)

    H_blocks = int(np.ceil(h / config.BLOCK_SIZE))
    W_blocks = int(np.ceil(w / config.BLOCK_SIZE))

    orient_blocks = np.zeros((H_blocks, W_blocks), dtype=np.float32)
    reliability = np.zeros((H_blocks, W_blocks), dtype=np.float32)

    for by in range(H_blocks):
        y0 = by * config.BLOCK_SIZE
        y1 = min((by + 1) * config.BLOCK_SIZE, h)
        for bx in range(W_blocks):
            x0 = bx * config.BLOCK_SIZE
            x1 = min((bx + 1) * config.BLOCK_SIZE, w)

            sx = Vx_s[y0:y1, x0:x1].sum()
            sy = Vy_s[y0:y1, x0:x1].sum()
            se = energy_s[y0:y1, x0:x1].sum()

            reliability[by, bx] = se / ((y1 - y0) * (x1 - x0) + 1e-12)
            if reliability[by, bx] < config.ENERGY_THRESHOLD:
                orient_blocks[by, bx] = 0.0
            else:
                orient_blocks[by, bx] = 0.5 * math.atan2(sx, sy)

    if reliability.max() > 0:
        reliability /= reliability.max()

    orient_img = np.kron(orient_blocks, np.ones((config.BLOCK_SIZE, config.BLOCK_SIZE), dtype=np.float32))
    orient_img = orient_img[:h, :w]

    return orient_blocks, orient_img, reliability


def visualize_orientation_map(img: np.ndarray,
                              orient_blocks: np.ndarray,
                              reliability: np.ndarray = None) -> np.ndarray:
    """Sovrappone la mappa di orientamento all'immagine."""
    vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    H_blocks, W_blocks = orient_blocks.shape

    for by in range(H_blocks):
        for bx in range(W_blocks):
            angle = float(orient_blocks[by, bx])
            if reliability is not None and reliability[by, bx] < config.REL_THRESHOLD:
                continue
            cx = int(bx * config.BLOCK_SIZE + config.BLOCK_SIZE / 2)
            cy = int(by * config.BLOCK_SIZE + config.BLOCK_SIZE / 2)
            dx = int(round(config.VIS_SCALE * math.cos(angle)))
            dy = int(round(config.VIS_SCALE * math.sin(angle)))
            cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 1, cv2.LINE_AA)

    return vis


# ================================================
# THINNING + CLEANING
# ================================================
def thinning_and_cleaning(binary_img: np.ndarray) -> np.ndarray:
    """Scheletrizza la binaria e rimuove piccoli oggetti."""
    mask = (binary_img > 0).astype(bool)
    mask = remove_small_objects(mask, min_size=config.MIN_OBJ_SIZE)
    mask = remove_small_holes(mask, area_threshold=config.MAX_HOLE_SIZE)
    skeleton = skeletonize(mask)

    return (skeleton > 0).astype(np.uint8) * 255


# ================================================
# HIGH-LEVEL PIPELINE
# ================================================
def preprocess_fingerprint(img: np.ndarray, debug_dir: Optional[str] = None):
    """Esegue la pipeline completa di pre-processing."""
    try:
        normalized = normalize_image(img)
        _debug_write(debug_dir, "normalized.png", normalized)

        denoised = denoise_image(normalized)
        _debug_write(debug_dir, "denoised.png", denoised)

        segmented = segment_fingerprint(denoised)
        _debug_write(debug_dir, "segmented.png", segmented)

        binary = robust_binarize(segmented)
        _debug_write(debug_dir, "binary.png", binary)

        skeleton = thinning_and_cleaning(binary)
        _debug_write(debug_dir, "skeleton.png", skeleton)

        orient_blocks, orient_img, reliability = compute_orientation_map(segmented)
        orientation_vis = visualize_orientation_map(segmented, orient_blocks, reliability)
        _debug_write(debug_dir, "orientation_map.png", orientation_vis)

        return {
            "normalized": normalized,
            "denoised": denoised,
            "segmented": segmented,
            "binary": binary,
            "skeleton": skeleton,
            "orientation_map": orient_img,
            "orientation_vis": orientation_vis,
        }

    except Exception as e:
        raise RuntimeError(f"preprocess_fingerprint failed: {e}") from e
