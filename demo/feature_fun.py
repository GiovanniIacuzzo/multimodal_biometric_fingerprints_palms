# demo/feature_fun.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ------------------------------------------------
# Normalizzazione
# ------------------------------------------------
def normalize(img):
    """Normalizza intensità [0,255] → [0,1]."""
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

# ------------------------------------------------
# Auto-rotazione in step di 90°
# ------------------------------------------------
def auto_rotate_90(img):
    """
    Ruota immagine in multipli di 90° e sceglie la rotazione
    con la massima 'coerenza' del campo di orientazione (std angle più alto).
    Utile per correggere rotazioni approssimative di 90°/180°.
    """
    best_score = -np.inf
    best_img = img
    for k in range(4):
        rotated = np.rot90(img, k)
        ang = compute_orientation_field(rotated)
        score = classify_std_angle(ang)
        if score > best_score:
            best_score = score
            best_img = rotated
    return best_img

# ------------------------------------------------
# Campo di orientazione
# ------------------------------------------------
def compute_orientation_field(img, block_size=16):
    """Calcola campo di orientamento su blocchi (in radianti)."""
    gy, gx = np.gradient(img)
    height, width = img.shape
    n_h = max(1, height // block_size)
    n_w = max(1, width // block_size)
    angles = np.zeros((n_h, n_w), dtype=np.float32)
    for bi in range(n_h):
        for bj in range(n_w):
            i = bi * block_size
            j = bj * block_size
            region_gx = gx[i:i+block_size, j:j+block_size]
            region_gy = gy[i:i+block_size, j:j+block_size]
            if region_gx.size == 0 or region_gy.size == 0:
                angles[bi, bj] = 0.0
                continue
            vxx = np.mean(region_gx ** 2)
            vyy = np.mean(region_gy ** 2)
            vxy = 2 * np.mean(region_gx * region_gy)
            theta = 0.5 * np.arctan2(vxy, (vxx - vyy + 1e-8))
            angles[bi, bj] = theta
    return angles

def classify_std_angle(angles):
    """Restituisce deviazione standard dell'orientazione (in gradi)."""
    ang = np.degrees(angles.flatten())
    ang = ang[~np.isnan(ang)]
    if ang.size == 0:
        return 0.0
    return float(np.std(ang))

# ------------------------------------------------
# Feature rotation-invariant
# ------------------------------------------------
def extract_rotation_invariant_features(img):
    """
    Estrae feature robuste alla rotazione:
    - LBP rotation-invariant (histogram)
    - statistiche globali del campo di orientazione (std, circular coherence)
    - opzionale: HOG globalizzato (ma HOG è sensibile alla rotazione)
    """
    # img expected in [0,1] float
    if img.max() > 1.0:
        img = normalize(img)

    # 1) LBP rotation invariant (uniform)
    P = 8
    R = 1
    lbp = local_binary_pattern((img * 255).astype(np.uint8), P=P, R=R, method='uniform')
    # bins: uniform patterns produce P+2 bins
    n_bins = P + 2
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, n_bins + 1), density=True)
    hist_lbp = hist_lbp.astype(np.float32)

    # 2) Orientation statistics
    angles = compute_orientation_field(img)
    std_angle = classify_std_angle(angles)
    ang_flat = np.degrees(angles.flatten())
    ang_flat = ang_flat[~np.isnan(ang_flat)]
    circ_coh = float(np.mean(np.cos(np.deg2rad(ang_flat)))) if ang_flat.size > 0 else 0.0

    # 3) Optional poche statistiche globali dell'immagine (contrast, energy)
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))

    feats = np.concatenate([[std_angle, circ_coh, mean_intensity, std_intensity], hist_lbp], axis=0)
    return feats.astype(np.float32)

# ------------------------------------------------
# Mantengo estrazione multi-scale HOG come compatibilità
# ------------------------------------------------
from skimage.feature import hog as sk_hog

def extract_multi_scale_features(img, scales=(1.0, 0.75, 0.5)):
    """HOG multi-scale (compatibilità con versione precedente)."""
    feats = []
    for s in scales:
        h, w = int(img.shape[1] * s), int(img.shape[0] * s)
        if h < 16 or w < 16:
            continue
        resized = cv2.resize((img * 255).astype(np.uint8), (h, w))
        h_feat = sk_hog(resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                        orientations=9, visualize=False, feature_vector=True)
        feats.append(h_feat.astype(np.float32))
    if feats:
        return np.concatenate(feats, axis=0)
    else:
        # fallback: histogram of intensities
        hist, _ = np.histogram(img, bins=32, range=(0,1), density=True)
        return hist.astype(np.float32)
