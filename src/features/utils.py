import os
import cv2
import numpy as np
from skimage.morphology import thin
import logging
from config import config

logger = logging.getLogger("minutiae_extractor")

# ===========================================
# UTILITY
# ===========================================
def ensure_dir(path: str):
    """Crea la directory se non esiste."""
    os.makedirs(path, exist_ok=True)


def save_debug_image(filename: str, img: np.ndarray, normalize: bool = True):
    """Salva immagine di debug in config.DEBUG_DIR, se attivo."""
    if not config.DEBUG_DIR:
        return

    ensure_dir(config.DEBUG_DIR)
    path = os.path.join(config.DEBUG_DIR, filename)

    try:
        if normalize:
            # Normalizza intensità per visualizzazione
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)
        logger.debug(f"Saved debug image: {path}")
    except Exception as e:
        logger.warning(f"Could not save debug image {filename}: {e}")


def maybe_invert(img: np.ndarray) -> np.ndarray:
    """Inverte se l’immagine è prevalentemente chiara (impronta chiara su sfondo scuro)."""
    return 255 - img if np.mean(img) > 127 else img


# ===========================================
# SKELETON CLEANING
# ===========================================
def compute_neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Conta i pixel vicini in una finestra 3×3 per ogni pixel del skeleton."""
    sk = (skel > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neigh = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neigh_no_center = neigh - sk
    return neigh_no_center.astype(np.uint8)


def clean_skeleton(img: np.ndarray, invert_auto: bool = True, prune_iters: int = None) -> np.ndarray:
    """
    Scheletrizza e pulisce la binaria.
    - `invert_auto`: inverte automaticamente se necessario
    - `prune_iters`: numero di iterazioni di pruning (default da config)
    """
    prune_iters = prune_iters or config.PRUNE_ITERS

    if invert_auto:
        img = maybe_invert(img)

    sk = (img > 0).astype(np.uint8)
    sk = thin(sk).astype(np.uint8)

    deg = compute_neighbor_count(sk)
    junction_mask = (deg >= 3)

    # Rimuove rami isolati o piccoli oggetti
    for _ in range(max(0, int(prune_iters))):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
        for lab in range(1, num):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area <= config.PRUNE_AREA:
                comp_mask = (labels == lab)
                if not np.any(junction_mask & comp_mask):
                    sk[comp_mask] = 0

    cleaned = (sk > 0).astype(np.uint8) * 255
    save_debug_image("skeleton_cleaned.png", cleaned)
    return cleaned


# ===========================================
# ORIENTATION MAP
# ===========================================
def compute_orientation_map(gray: np.ndarray, sigma: float = None):
    """
    Calcola la mappa di orientamento e la coerenza locale.
    Usa derivate Sobel e smoothing gaussiano.
    """
    sigma = sigma or config.ORIENT_SIGMA
    img = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    gxx, gyy, gxy = gx * gx, gy * gy, gx * gy

    Sxx = cv2.GaussianBlur(gxx, (0, 0), sigma)
    Syy = cv2.GaussianBlur(gyy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(gxy, (0, 0), sigma)

    orient = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)

    # Coerenza locale: misura l’affidabilità dell’orientamento
    lambda1 = 0.5 * (Sxx + Syy + np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2))
    lambda2 = 0.5 * (Sxx + Syy - np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2))
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    coherence = np.clip(coherence, 0, 1)

    save_debug_image("orientation_map.png", orient, normalize=True)
    save_debug_image("coherence_map.png", coherence, normalize=True)

    return orient, coherence
