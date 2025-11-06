import os
from typing import Optional
import numpy as np
import cv2
import math
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, skeletonize

# --- utility small logger
def _debug_write(debug_dir: Optional[str], name: str, img: np.ndarray):
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, name), img if img.dtype == np.uint8 else (img.astype(np.uint8) * 255))

# -------------------------
# NORMALIZATION + CLAHE
# -------------------------
def normalize_image(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    img_u8 = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_u8)

# -------------------------
# DENOISING
# -------------------------
def denoise_image(img: np.ndarray) -> np.ndarray:
    b = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=7)
    return cv2.GaussianBlur(b, (3, 3), 0.7)

# -------------------------
# BINARIZATION
# -------------------------
def robust_binarize(img: np.ndarray,
                    sauvola_win: int = 25,
                    sauvola_k: float = 0.2,
                    local_patch: int = 64,
                    min_size: int = 30,
                    max_hole: int = 100) -> np.ndarray:
    img_f = img.astype(np.uint8)
    thresh = threshold_sauvola(img_f, window_size=sauvola_win, k=sauvola_k)
    binary = img_f < thresh

    h, w = img_f.shape
    for i in range(0, h, local_patch):
        for j in range(0, w, local_patch):
            patch = img_f[i:i+local_patch, j:j+local_patch]
            if patch.size == 0:
                continue
            try:
                o = threshold_otsu(patch)
                patch_bin = patch < o
                binary[i:i+local_patch, j:j+local_patch] |= patch_bin
            except Exception:
                pass

    marker = binary.copy()
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marker = cv2.erode(marker.astype(np.uint8), se, iterations=1).astype(bool)
    mask = binary.astype(bool)
    recon = reconstruction(marker.astype(np.uint8), mask.astype(np.uint8), method='dilation')
    recon = recon.astype(bool)

    recon = remove_small_objects(recon, min_size=min_size)
    recon = remove_small_holes(recon, area_threshold=max_hole)

    return (recon > 0).astype(np.uint8) * 255

# -------------------------
# SEGMENTATION
# -------------------------
def segment_fingerprint(img: np.ndarray, min_area: int = 5000) -> np.ndarray:
    """
    Segmentazione in stile Viola–Jones:
    trova la regione dell'impronta e la ritaglia senza tentare di estrarre le ridge.
    """
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
    if cv2.contourArea(largest_contour) < min_area:
        return gray

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped = gray[y:y + h, x:x + w]
    return cropped

# -------------------------
# ORIENTATION
# -------------------------
def compute_orientation_map(img: np.ndarray,
                            block_size: int = 16,
                            sigma: float = 5.0,
                            energy_threshold: float = 1e-4):
    """
    Calcola una mappa di orientamento stabile per le ridge.
    Restituisce:
      - orient_blocks: (H_blocks, W_blocks) angoli in radianti (direzione delle ridge)
      - orient_img: (h, w) immagine degli angoli replicata per pixel
      - reliability: (H_blocks, W_blocks) valori normalizzati [0..1] di energia/coerenza
    Parametri:
      - block_size: dimensione del blocco in pixel
      - sigma: sigma gaussian smoothing applicato ai termini tensoriali
      - energy_threshold: soglia minima (sulla energia media del blocco) per considerarlo valido
    """
    # grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    h, w = gray.shape
    gray_f = gray.astype(np.float32) / 255.0

    # gradienti (Sobel)
    Gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    # termini tensoriali
    Vx = 2.0 * (Gx * Gy)        # numeratore  (2 Gx Gy)
    Vy = (Gx**2 - Gy**2)        # denominatore (Gx^2 - Gy^2)
    energy = (Gx**2 + Gy**2)    # energia locale

    # smoothing spaziale dei termini per stabilizzare
    Vx_s = gaussian_filter(Vx, sigma=sigma)
    Vy_s = gaussian_filter(Vy, sigma=sigma)
    energy_s = gaussian_filter(energy, sigma=sigma)

    # dimensione a blocchi (gestione bordi non multipli)
    H_blocks = int(np.ceil(h / block_size))
    W_blocks = int(np.ceil(w / block_size))

    orient_blocks = np.zeros((H_blocks, W_blocks), dtype=np.float32)
    reliability = np.zeros((H_blocks, W_blocks), dtype=np.float32)

    # calcolo angolo medio per blocco
    for by in range(H_blocks):
        y0 = by * block_size
        y1 = min((by + 1) * block_size, h)

        for bx in range(W_blocks):
            x0 = bx * block_size
            x1 = min((bx + 1) * block_size, w)

            sx = Vx_s[y0:y1, x0:x1].sum()
            sy = Vy_s[y0:y1, x0:x1].sum()
            se = energy_s[y0:y1, x0:x1].sum()

            # salva energia media del blocco (per reliability)
            reliability[by, bx] = se / ((y1 - y0) * (x1 - x0) + 1e-12)

            if reliability[by, bx] < energy_threshold:
                # blocco poco informativo -> lascia 0 angolo (invalid)
                orient_blocks[by, bx] = 0.0
            else:
                # theta = 0.5 * atan2( sum(2GxGy), sum(Gx^2 - Gy^2) )
                theta = 0.5 * math.atan2(sx, sy)
                # theta indica la direzione delle ridge
                orient_blocks[by, bx] = float(theta)

    # normalizza reliability su [0,1]
    max_rel = reliability.max()
    if max_rel > 0:
        reliability = reliability / (max_rel + 1e-12)

    # costruisci orient_img a risoluzione pixel (riempi blocchi)
    orient_img = np.zeros((h, w), dtype=np.float32)
    for by in range(H_blocks):
        y0 = by * block_size
        y1 = min((by + 1) * block_size, h)
        for bx in range(W_blocks):
            x0 = bx * block_size
            x1 = min((bx + 1) * block_size, w)
            orient_img[y0:y1, x0:x1] = orient_blocks[by, bx]

    return orient_blocks, orient_img, reliability

def visualize_orientation_map(img: np.ndarray,
                              orient_blocks: np.ndarray,
                              block_size: int = 16,
                              scale: int = 8,
                              reliability: np.ndarray = None,
                              draw_only_reliable: bool = True,
                              reliability_threshold: float = 0.2,
                              color: tuple = (0, 0, 255)) -> np.ndarray:
    """
    Disegna la mappa di orientamento (orient_blocks) sovrapposta all'immagine.
    - reliability: matrice (H_blocks, W_blocks) con valori normalizzati [0..1] (opzionale)
    - draw_only_reliable: se True ignora i blocchi con reliability < reliability_threshold
    - reliability_threshold: soglia per disegnare
    - color: BGR color for lines (default rosso)
    Restituisce immagine BGR uint8.
    """
    # Prepara immagine BGR
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
        if vis.dtype != np.uint8:
            vis = (vis.astype(np.uint8))

    H_blocks, W_blocks = orient_blocks.shape

    for by in range(H_blocks):
        for bx in range(W_blocks):
            angle = float(orient_blocks[by, bx])

            # se richiesta, salta blocchi poco affidabili
            if draw_only_reliable and (reliability is not None):
                rel = float(reliability[by, bx])
                if rel < reliability_threshold:
                    continue

            # centro del blocco (attenzione ai bordi)
            cx = int(bx * block_size + block_size / 2)
            cy = int(by * block_size + block_size / 2)

            # calcola segmento lungo la direzione delle ridge (theta)
            dx = int(round(scale * math.cos(angle)))
            dy = int(round(scale * math.sin(angle)))

            pt1 = (max(0, cx - dx), max(0, cy - dy))
            pt2 = (min(vis.shape[1] - 1, cx + dx), min(vis.shape[0] - 1, cy + dy))

            cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)

    # opzionale: sovrapponi heatmap di reliability per debugging (commenta se non vuoi)
    if reliability is not None:
        # ridimensiona reliability (blocchi -> immagine) per overlay
        rel_img = np.kron(reliability, np.ones((block_size, block_size), dtype=np.float32))
        rel_img = rel_img[:vis.shape[0], :vis.shape[1]]
        rel_vis = (np.clip(rel_img, 0.0, 1.0) * 255).astype(np.uint8)
        rel_col = cv2.applyColorMap(rel_vis, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.8, rel_col, 0.2, 0)

    return vis

# -------------------------
# THINNING + CLEANING
# -------------------------
def thinning_and_cleaning(binary_img: np.ndarray,
                          min_object_size: int = 30,
                          max_hole_size: int = 20,
                          smooth_edges: bool = True) -> np.ndarray:
    mask = (binary_img > 0).astype(bool)
    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = remove_small_holes(mask, area_threshold=max_hole_size)
    skeleton = skeletonize(mask)

    if smooth_edges:
        sk_uint8 = skeleton.astype(np.uint8)
        h, w = sk_uint8.shape
        cleaned = sk_uint8.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if sk_uint8[y, x] == 1:
                    n = np.sum(sk_uint8[y-1:y+2, x-1:x+2]) - 1
                    if n <= 1:
                        cleaned[y, x] = 0
        skeleton = cleaned.astype(bool)

    return (skeleton > 0).astype(np.uint8) * 255

# -------------------------
# HIGH-LEVEL PIPELINE
# -------------------------
def preprocess_fingerprint(img: np.ndarray, debug_dir: Optional[str] = None):
    """
    Esegue pipeline completa di pre-processing dell'impronta digitale:
    - Normalizzazione e miglioramento contrasto
    - Denoising
    - Segmentazione e ritaglio automatico (scontorno)
    - Binarizzazione robusta
    - Thinning (scheletrizzazione) e pulizia
    - Calcolo e visualizzazione mappa di orientamento delle ridge
    """
    try:
        # 1. Normalizzazione e miglioramento contrasto
        normalized = normalize_image(img)
        _debug_write(debug_dir, "normalized.png", normalized)

        # 2. Denoising
        denoised = denoise_image(normalized)
        _debug_write(debug_dir, "denoised.png", denoised)

        # 3. Segmentazione + ritaglio automatico
        segmented = segment_fingerprint(denoised)
        _debug_write(debug_dir, "segmented.png", segmented)

        # Salva anche la versione ritagliata separatamente
        if debug_dir:
            cropped_path = os.path.join(debug_dir, "fingerprint_cropped.png")
            cv2.imwrite(cropped_path, segmented)

        # 4. Binarizzazione robusta
        binary = robust_binarize(segmented)
        _debug_write(debug_dir, "binary_raw.png", binary)

        # 5. Thinning e pulizia finale
        skeleton = thinning_and_cleaning(binary, min_object_size=15, max_hole_size=20, smooth_edges=True)
        _debug_write(debug_dir, "skeleton.png", skeleton)

        # 6. Calcolo della mappa di orientamento
        orient_blocks, orient_img, reliability = compute_orientation_map(segmented, block_size=16, sigma=7.0, energy_threshold=1e-2)
        orientation_vis = visualize_orientation_map(segmented, orient_blocks, block_size=16, scale=8, reliability=reliability, draw_only_reliable=True)


        # Salva la mappa di orientamento
        _debug_write(debug_dir, "orientation_map.png", orientation_vis)

        # Output finale completo
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
