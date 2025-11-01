import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from config import *

# =====================================================
# NORMALIZZAZIONE DI BASE
# =====================================================
def normalize(img):
    """
    Normalizza l'immagine in scala [0, 255].
    Riduce variazioni di illuminazione e contrasto.
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# =====================================================
# GABOR FEATURES
# =====================================================
def build_gabor_kernels(thetas=np.arange(0, np.pi, np.pi/8), k_sizes=(11, 21)):
    """Precalcola i kernel di Gabor per efficienza."""
    kernels = []
    for theta in thetas:
        for ksize in k_sizes:
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=4.0,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernels.append(kernel)
    return kernels


# Kernel Gabor precalcolati globalmente
GABOR_KERNELS = build_gabor_kernels()


def gabor_features(img):
    """
    Estrae media e varianza delle risposte ai filtri di Gabor precalcolati.
    Restituisce un vettore [mean, std] per ciascun kernel.
    """
    feats = []
    for kernel in GABOR_KERNELS:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        mean, std = cv2.meanStdDev(filtered)
        feats.extend([mean[0][0], std[0][0]])
    return np.array(feats, dtype=np.float32)


# =====================================================
# LBP FEATURES
# =====================================================
def lbp_features(img, P=8, R=1):
    """
    Estrae l'istogramma normalizzato LBP (Local Binary Pattern).
    """
    lbp = local_binary_pattern(img, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


# =====================================================
# MULTI-SCALE FEATURE EXTRACTION
# =====================================================
def extract_multi_scale_features(img):
    """
    Estrae e concatena feature combinate (HOG, Gabor, LBP)
    su più scale definite in config. Ogni blocco di feature è
    normalizzato internamente (L2). La standardizzazione
    globale viene fatta successivamente su tutto il dataset.
    """
    all_feats = []

    for size in IMG_SCALES:
        img_resized = cv2.resize(img, size)
        feats = []

        # HOG
        if USE_HOG:
            hog_feat = hog(
                img_resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                transform_sqrt=True,
                feature_vector=True,
            )
            hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-6)
            feats.append(hog_feat.astype(np.float32))

        # GABOR
        if USE_GABOR:
            gab = gabor_features(img_resized)
            gab = gab / (np.linalg.norm(gab) + 1e-6)
            feats.append(gab.astype(np.float32))

        # LBP
        if USE_LBP:
            lbp = lbp_features(img_resized)
            lbp = lbp / (np.linalg.norm(lbp) + 1e-6)
            feats.append(lbp.astype(np.float32))

        if feats:
            all_feats.append(np.concatenate(feats))

    if not all_feats:
        return np.zeros((1,), dtype=np.float32)

    return np.concatenate(all_feats).astype(np.float32)


# =====================================================
# STANDARDIZZAZIONE GLOBALE
# =====================================================
def standardize_global_features(features_dict):
    """
    Applica StandardScaler globale a tutte le feature del dataset.
    Modifica in-place il dizionario features_dict.
    """
    # Costruzione matrice completa
    all_feats = np.vstack([f[1] for cls in features_dict.values() for f in cls])
    scaler = StandardScaler().fit(all_feats)

    for cls, img_feats_list in features_dict.items():
        for i, (path, feat) in enumerate(img_feats_list):
            scaled_feat = scaler.transform(feat.reshape(1, -1)).flatten()
            features_dict[cls][i] = (path, scaled_feat)

    return features_dict, scaler


# =====================================================
# ORIENTATION FIELD
# =====================================================
def compute_orientation_field(img):
    """
    Calcola il campo di orientamento locale dell'immagine.
    Filtra rumore e bilancia localmente gx, gy.
    """
    img_blur = cv2.GaussianBlur(img, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)
    gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    h, w = img.shape
    Hb, Wb = h // BLOCK_SIZE, w // BLOCK_SIZE
    angles = np.zeros((Hb, Wb), dtype=np.float32)

    for i in range(Hb):
        for j in range(Wb):
            y0, x0 = i * BLOCK_SIZE, j * BLOCK_SIZE
            gx_blk = gx[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            gy_blk = gy[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]

            gx_blk -= gx_blk.mean()
            gy_blk -= gy_blk.mean()

            Vx = 2 * np.sum(gx_blk * gy_blk)
            Vy = np.sum(gx_blk**2 - gy_blk**2)
            angles[i, j] = 0.5 * np.arctan2(Vx, Vy)

    # Lisciamento finale del campo angolare
    return cv2.GaussianBlur(angles, (3, 3), 0)


# =====================================================
# VARIABILITÀ ANGOLARE (per classificazione grossolana)
# =====================================================
def classify_std_angle(angles):
    """
    Calcola la deviazione standard delle angolazioni (in gradi).
    """
    ang_deg = np.degrees(angles).flatten()
    ang_deg = ang_deg[~np.isnan(ang_deg)]
    return float(np.std(ang_deg))
