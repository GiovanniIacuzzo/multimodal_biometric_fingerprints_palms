import numpy as np
import cv2
from typing import Optional
from scipy.ndimage import gaussian_filter

# ================================================
# ORIENTATION MAP
# ================================================
def compute_orientation_map(img: np.ndarray,
                            block_size: int = 16,
                            smooth_sigma: float = 3.0,
                            invert_if_needed: bool = True,
                            smooth_orientation_sigma: float = 3.0,
                            mask: Optional[np.ndarray] = None):
    """
    Calcola il campo di orientamento ridotto a blocchi, filtrando con una mask opzionale.
    """
    # --- Preprocessing ---
    if img.dtype == np.uint8:
        f = img.astype(np.float32) / 255.0
    else:
        f = img.astype(np.float32)
        if f.max() > 1.0 or f.min() < 0.0:
            f = (f - f.min()) / (f.max() - f.min() + 1e-12)

    if invert_if_needed:
        if np.mean(f[f > np.median(f)]) > np.mean(f[f <= np.median(f)]):
            f = 1.0 - f

    f_s = gaussian_filter(f, sigma=max(0.5, smooth_sigma / 2.0))

    # Gradienti
    Gx = cv2.Sobel((f_s * 255).astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel((f_s * 255).astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    Gxx = gaussian_filter(Gx * Gx, sigma=smooth_sigma)
    Gyy = gaussian_filter(Gy * Gy, sigma=smooth_sigma)
    Gxy = gaussian_filter(Gx * Gy, sigma=smooth_sigma)

    reliability = np.sqrt((Gxx - Gyy) ** 2 + 4.0 * Gxy ** 2)
    rmin, rmax = np.percentile(reliability, [2, 98])
    reliability = np.clip((reliability - rmin) / (rmax - rmin + 1e-12), 0.0, 1.0)

    orientation = 0.5 * np.arctan2(2.0 * Gxy, (Gxx - Gyy) + 1e-12)
    orientation = orientation + np.pi / 2.0

    h, w = f.shape
    n_by, n_bx = h // block_size, w // block_size
    orient_blocks = np.zeros((n_by, n_bx), dtype=np.float32)
    rel_blocks = np.zeros((n_by, n_bx), dtype=np.float32)

    for by in range(n_by):
        for bx in range(n_bx):
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size

            if mask is not None:
                submask = mask[y0:y1, x0:x1]
                # salta blocchi fuori area impronta
                if np.mean(submask > 0) < 0.3:
                    continue

            block_theta = orientation[y0:y1, x0:x1]
            block_r = reliability[y0:y1, x0:x1]
            if block_theta.size == 0:
                continue

            wts = block_r.flatten() + 1e-6
            s = np.sum(wts * np.sin(2.0 * block_theta).flatten())
            c = np.sum(wts * np.cos(2.0 * block_theta).flatten())
            orient_blocks[by, bx] = 0.5 * np.arctan2(s, c)
            rel_blocks[by, bx] = np.mean(block_r)

    # Smoothing direzionale
    sin2 = np.sin(2.0 * orient_blocks)
    cos2 = np.cos(2.0 * orient_blocks)
    sin2_s = gaussian_filter(sin2, sigma=smooth_orientation_sigma)
    cos2_s = gaussian_filter(cos2, sigma=smooth_orientation_sigma)
    orient_blocks = 0.5 * np.arctan2(sin2_s, cos2_s)

    orient_img = cv2.resize(orient_blocks, (w, h), interpolation=cv2.INTER_LINEAR)
    rel_img = cv2.resize(rel_blocks, (w, h), interpolation=cv2.INTER_LINEAR)
    orient_img = (orient_img + np.pi / 2) % np.pi - np.pi / 2

    return orient_blocks, orient_img, rel_img

def visualize_orientation(img: np.ndarray,
                          orient_img: np.ndarray,
                          reliability_img: np.ndarray = None,
                          block_size: int = 16,
                          scale: int = 8,
                          rel_thresh: float = 0.2,
                          mask: Optional[np.ndarray] = None,
                          color=(0, 0, 255)):
    """
    Visualizza il campo di orientamento con filtraggio tramite mask.
    """
    if len(img.shape) == 2:
        vis = cv2.cvtColor((np.clip(img, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    h, w = orient_img.shape
    step = block_size
    half = step // 2

    for by in range(0, h // step):
        for bx in range(0, w // step):
            cy = int(by * step + half)
            cx = int(bx * step + half)
            if cy >= h or cx >= w:
                continue

            if mask is not None and mask[cy, cx] == 0:
                continue  # ignora sfondo

            if reliability_img is not None and reliability_img[cy, cx] < rel_thresh:
                continue

            angle = orient_img[cy, cx]
            dx = int(round(scale * np.cos(angle)))
            dy = int(round(scale * np.sin(angle)))
            x1, y1 = max(0, cx - dx), max(0, cy - dy)
            x2, y2 = min(w - 1, cx + dx), min(h - 1, cy + dy)

            cv2.line(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    gray2bgr = cv2.cvtColor((np.clip(img, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(vis, 0.8, gray2bgr, 0.2, 0)
    return overlay