import os
from typing import Optional, Dict
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_sauvola, threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, skeletonize
from scipy.ndimage import gaussian_filter
from config import config

# ================================================
# NORMALIZATION + CLAHE
# ================================================
def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalizza immagine e applica CLAHE in modo robusto.
    Accetta input float [0,1] o uint8.
    """
    # Convert to float in [0,1]
    if img.dtype == np.uint8:
        f = img.astype(np.float32) / 255.0
    else:
        f = img.astype(np.float32)
        # se range non [0,1], normalizziamo
        if f.max() > 1.0 or f.min() < 0.0:
            f = (f - f.min()) / (f.max() - f.min() + 1e-12)

    # Stretch contrast (robusto)
    f = (f - np.percentile(f, 1)) / (np.percentile(f, 99) - np.percentile(f, 1) + 1e-12)
    f = np.clip(f, 0.0, 1.0)

    # CLAHE richiede uint8
    img_u8 = (f * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=getattr(config, "CLAHE_CLIP_LIMIT", 2.0),
        tileGridSize=(getattr(config, "CLAHE_TILE_SIZE", 8), getattr(config, "CLAHE_TILE_SIZE", 8))
    )
    out = clahe.apply(img_u8)
    return out

# ================================================
# DENOISING
# ================================================
def denoise_image(img: np.ndarray) -> np.ndarray:
    """Riduzione rumore con filtro bilaterale + Gaussiano."""
    b = cv2.bilateralFilter(
        img,
        d=getattr(config, "BILATERAL_D", 5),
        sigmaColor=getattr(config, "BILATERAL_SIGMA_COLOR", 75),
        sigmaSpace=getattr(config, "BILATERAL_SIGMA_SPACE", 75)
    )
    return cv2.GaussianBlur(b, (3, 3), getattr(config, "GAUSSIAN_SIGMA", 0.5))

# ================================================
# BINARIZATION
# ================================================
def binarize(img: np.ndarray) -> np.ndarray:
    """
    Binarizza combinando Sauvola globale + local thresholding più veloce.
    Restituisce immagine uint8 (0/255).
    """
    img_f = img.astype(np.uint8)

    # 1) Sauvola (map)
    try:
        sauv = threshold_sauvola(img_f, window_size=getattr(config, "SAUVOLA_WIN", 25), k=getattr(config, "SAUVOLA_K", 0.2))
        binary_sauv = img_f < sauv
    except Exception:
        # fallback: threshold locale di skimage (più veloce in alcune config)
        sauv = threshold_local(img_f, block_size=getattr(config, "SAUVOLA_WIN", 25))
        binary_sauv = img_f < sauv

    # 2) Patch-wise Otsu ma su downsample/stride per velocizzare
    h, w = img_f.shape
    patch = getattr(config, "LOCAL_PATCH", 64)
    binary = binary_sauv.copy()

    # Se l'immagine è piccola, usa Otsu globale
    if h * w <= patch * patch:
        try:
            o = threshold_otsu(img_f)
            binary |= (img_f < o)
        except Exception:
            pass
    else:
        # Scorri con step = patch (non sovrapposto) per velocità
        for i in range(0, h, patch):
            i1 = min(i + patch, h)
            for j in range(0, w, patch):
                j1 = min(j + patch, w)
                sub = img_f[i:i1, j:j1]
                if sub.size == 0:
                    continue
                # se troppi pochi livelli, salta
                if sub.std() < 2.0:
                    # se molto uniforme usa soglia media
                    o = sub.mean()
                    binary[i:i1, j:j1] |= (sub < o)
                    continue
                try:
                    o = threshold_otsu(sub)
                    binary[i:i1, j:j1] |= (sub < o)
                except Exception:
                    # fallback: media locale
                    o = sub.mean()
                    binary[i:i1, j:j1] |= (sub < o)

    # 3) Morphological reconstruction per pulire (usa booleani)
    marker = cv2.erode(binary.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1).astype(bool)
    mask = binary.astype(bool)
    recon = reconstruction(marker.astype(bool), mask.astype(bool), method='dilation')
    recon = remove_small_objects(recon.astype(bool), min_size=getattr(config, "MIN_OBJ_SIZE", 64))
    recon = remove_small_holes(recon.astype(bool), area_threshold=getattr(config, "MAX_HOLE_SIZE", 64))

    return (recon > 0).astype(np.uint8) * 255

# ================================================
# SEGMENTATION
# ================================================
def segment_fingerprint(img: np.ndarray, debug_dir: Optional[str] = None) -> (np.ndarray, np.ndarray):
    """
    Segmentazione robusta dell'impronta con generazione della maschera.
    Restituisce:
      - cropped: impronta ritagliata
      - cropped_mask: maschera binaria (255=impronta, 0=sfondo)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=getattr(config, "CLAHE_CLIP_LIMIT", 2.0),
                            tileGridSize=(getattr(config, "CLAHE_TILE_SIZE", 8),
                                          getattr(config, "CLAHE_TILE_SIZE", 8)))
    stab = clahe.apply(gray)
    blur = cv2.GaussianBlur(stab, (5, 5), 0)

    # Threshold Otsu
    try:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        _, mask = cv2.threshold(blur, int(np.mean(blur)), 255, cv2.THRESH_BINARY)

    # Se foreground è chiaro -> invertiamo
    fg_mean = np.mean(gray[mask == 255]) if np.any(mask == 255) else 0
    bg_mean = np.mean(gray[mask == 0]) if np.any(mask == 0) else 0
    if fg_mean > bg_mean:
        mask = cv2.bitwise_not(mask)

    # Morfologia per regioni compatte
    k = getattr(config, "SEG_KERNEL_SIZE", 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Contorno più grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, np.ones_like(gray, dtype=np.uint8) * 255

    contours = [c for c in contours if cv2.contourArea(c) >= getattr(config, "MIN_SEGMENT_AREA", 500)]
    if not contours:
        return gray, np.ones_like(gray, dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    mask_hull = np.zeros_like(mask)
    cv2.drawContours(mask_hull, [hull], -1, 255, thickness=-1)

    # Crop con margine
    x, y, w_box, h_box = cv2.boundingRect(hull)
    margin = getattr(config, "SEG_CROP_MARGIN", 10)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(gray.shape[1], x + w_box + margin), min(gray.shape[0], y + h_box + margin)

    cropped = gray[y0:y1, x0:x1]
    hull_crop = mask_hull[y0:y1, x0:x1]
    cropped_mask = (hull_crop > 0).astype(np.uint8) * 255

    # Applica maschera per pulizia bordo
    cropped = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "segment_cropped.png"), cropped)
        cv2.imwrite(os.path.join(debug_dir, "segment_mask.png"), cropped_mask)

    return cropped, cropped_mask

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

# ================================================
# THINNING + CLEANING
# ================================================
def thinning_and_cleaning(binary_img: np.ndarray) -> np.ndarray:
    """Scheletizza la binaria e rimuove piccoli oggetti (usa booleani)."""
    mask = (binary_img > 0).astype(bool)
    mask = remove_small_objects(mask, min_size=getattr(config, "MIN_OBJ_SIZE", 64))
    mask = remove_small_holes(mask, area_threshold=getattr(config, "MAX_HOLE_SIZE", 64))
    skeleton = skeletonize(mask)  # expects boolean
    return (skeleton > 0).astype(np.uint8) * 255

# ================================================
# PIPELINE
# ================================================
def preprocess_fingerprint(img: np.ndarray, debug_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Pipeline completa per il preprocessing di un'impronta digitale.
    Include:
        - Normalizzazione e denoising
        - Segmentazione e maschera foreground
        - Binarizzazione e thinning
        - Calcolo campo di orientamento con maschera
        - Visualizzazione orientamento (solo area impronta)
    """
    try:
        # 1. Normalizzazione e rimozione rumore
        normalized = normalize_image(img)
        denoised = denoise_image(normalized)

        # 2. Segmentazione con restituzione della maschera
        segmented, mask = segment_fingerprint(denoised, debug_dir=debug_dir)

        # 3. Binarizzazione e thinning (scheletro)
        binary = binarize(segmented)
        skeleton = thinning_and_cleaning(binary)

        # 4. Calcolo campo di orientamento vincolato alla maschera
        orient_blocks, orient_img, reliability = compute_orientation_map(
            segmented,
            block_size=16,
            smooth_sigma=3.0,
            invert_if_needed=True,
            smooth_orientation_sigma=3.0,
            mask=mask
        )

        # 5. Visualizzazione del campo di orientamento
        orientation_vis = visualize_orientation(
            img=segmented,
            orient_img=orient_img,
            reliability_img=reliability,
            block_size=16,
            scale=7,
            rel_thresh=0.1,
            mask=mask
        )

        # 6. Salvataggio debug (se richiesto)
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "normalized.png"), normalized)
                cv2.imwrite(os.path.join(debug_dir, "denoised.png"), denoised)
                cv2.imwrite(os.path.join(debug_dir, "segmented.png"), segmented)
                cv2.imwrite(os.path.join(debug_dir, "mask.png"), mask)
                cv2.imwrite(os.path.join(debug_dir, "binary.png"), binary)
                cv2.imwrite(os.path.join(debug_dir, "skeleton.png"), skeleton)
                cv2.imwrite(os.path.join(debug_dir, "orientation_vis.png"), orientation_vis)
            except Exception:
                pass  # non bloccare la pipeline se il salvataggio fallisce

        # 7. Output strutturato
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
