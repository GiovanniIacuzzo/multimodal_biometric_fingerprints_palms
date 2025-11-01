from skimage.feature import hog, local_binary_pattern
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import *

# -------------------------------
# NORMALIZZAZIONE
# -------------------------------
def normalize(img):
    """
    Normalizza l'immagine in scala 0-255.
    Utile per ridurre l'impatto di variazioni di illuminazione.
    """
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -------------------------------
# GABOR FEATURES
# -------------------------------
def gabor_features(img, thetas=np.arange(0, np.pi, np.pi/8), k_sizes=[11,21]):
    """
    Estrae media e varianza delle risposte a filtri di Gabor
    su più orientamenti (theta) e frequenze (k_sizes).
    """
    feats = []
    for theta in thetas:
        for ksize in k_sizes:
            kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            feats.append(filtered.mean())
            feats.append(filtered.var())
    return np.array(feats)

# -------------------------------
# LBP FEATURES
# -------------------------------
def lbp_features(img, P=8, R=1):
    """
    Estrae la rappresentazione LBP dell'immagine e calcola l'istogramma normalizzato.
    LBP cattura micro-strutture locali utili per l'identificazione di impronte.
    """
    lbp = local_binary_pattern(img, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

# -------------------------------
# HOG + GABOR + LBP MULTI-SCALE
# -------------------------------
def extract_multi_scale_features(img):
    """
    Estrae feature combinate (HOG, Gabor, LBP) su più scale.
    Standardizza ogni feature prima di concatenarla.
    """
    all_feats = []
    for size in IMG_SCALES:
        img_resized = cv2.resize(img, size)
        feats = []

        if USE_HOG:
            hog_feat = hog(img_resized, orientations=9, pixels_per_cell=(8,8),
                           cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
            feats.append(hog_feat)
        if USE_GABOR:
            feats.append(gabor_features(img_resized))
        if USE_LBP:
            feats.append(lbp_features(img_resized))

        # Standardizzazione delle feature
        if feats:
            feats = [StandardScaler().fit_transform(f.reshape(-1,1)).flatten() for f in feats]
            all_feats.extend(feats)
    return np.concatenate(all_feats)

# -------------------------------
# ORIENTATION FIELD
# -------------------------------
def compute_orientation_field(img):
    """
    Calcola il campo di orientamento locale dell'immagine.
    Utile per classificare pattern globali Arch/Loop/Whorl.
    """
    img_blur = cv2.GaussianBlur(img, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)
    gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    h, w = img.shape
    Hb, Wb = h // BLOCK_SIZE, w // BLOCK_SIZE
    angles = np.zeros((Hb, Wb))

    for i in range(Hb):
        for j in range(Wb):
            y0, x0 = i*BLOCK_SIZE, j*BLOCK_SIZE
            gx_blk = gx[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE]
            gy_blk = gy[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE]

            Vx = 2 * np.sum(gx_blk * gy_blk)
            Vy = np.sum(gx_blk**2 - gy_blk**2)
            angles[i,j] = 0.5 * np.arctan2(Vx, Vy)

    return cv2.GaussianBlur(angles, (3,3), 0)

def classify_std_angle(angles):
    """
    Restituisce la deviazione standard delle angolazioni in gradi.
    Valore basso => pattern semplice (Arch), alto => pattern complesso (Whorl).
    """
    ang_deg = np.degrees(angles).flatten()
    ang_deg = ang_deg[~np.isnan(ang_deg)]
    return np.std(ang_deg)
