# demo/feature_fun.py
import cv2
import numpy as np

# =====================================================
# NORMALIZZAZIONE
# =====================================================
def normalize(img):
    """Normalizza intensità [0,255] → [0,1]."""
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

# =====================================================
# ORIENTATION FIELD + FEATURES
# =====================================================
def compute_orientation_field(img, block_size=16):
    """Calcola campo di orientamento su blocchi."""
    gy, gx = np.gradient(img)
    height, width = img.shape
    angles = np.zeros((height // block_size, width // block_size), dtype=np.float32)
    for i in range(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            region_gx = gx[i:i+block_size, j:j+block_size]
            region_gy = gy[i:i+block_size, j:j+block_size]
            vxx = np.mean(region_gx ** 2)
            vyy = np.mean(region_gy ** 2)
            vxy = 2 * np.mean(region_gx * region_gy)
            theta = 0.5 * np.arctan2(vxy, (vxx - vyy + 1e-5))
            angles[i // block_size, j // block_size] = theta
    return angles

def orientation_features(angles):
    """
    Restituisce vettore di feature dall'orientation field:
    [std_deg, circular_coherence, median_abs_angle, energy]
    """
    ang = np.degrees(angles).flatten()
    ang = ang[~np.isnan(ang)]
    if ang.size == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    std_deg = float(np.std(ang))
    circ_coh = float(np.mean(np.cos(np.radians(ang))))
    med = float(np.median(np.abs(ang)))
    energy = float(np.mean(np.abs(ang)))
    return np.array([std_deg, circ_coh, med, energy], dtype=np.float32)

def classify_std_angle(angles):
    """Compatibilità con vecchie funzioni."""
    return float(orientation_features(angles)[0])

# =====================================================
# MULTI-SCALE FEATURE (esempio HOG + Gabor)
# =====================================================
from skimage.feature import hog

def extract_multi_scale_features(img):
    """Combina HOG e Gabor su più scale."""
    feats = []
    scales = [1.0, 0.75, 0.5]
    for s in scales:
        resized = cv2.resize(img, None, fx=s, fy=s)
        h = hog(resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                orientations=9, visualize=False, feature_vector=True)
        feats.append(h.astype(np.float32))
    feats = np.concatenate(feats, axis=0)
    return feats

def standardize_global_features(features_dict):
    """
    Applica StandardScaler globale in modo robusto a tutte le feature del dataset.

    - features_dict: dict {classe: [(path, feat_array), ...]}
    - Restituisce: (features_dict_scaled, scaler)
        - features_dict_scaled: lo stesso dizionario con le feature scalate (in-place)
        - scaler: oggetto StandardScaler usato (o None se non applicabile)

    Nota:
    - Gestisce feature di lunghezza variabile effettuando padding alla lunghezza massima.
    - Se ci sono meno di 2 campioni totali non applica lo scaling e restituisce (features_dict, None).
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Raccogli tutte le feature
    all_feats = []
    for cls, lst in features_dict.items():
        for _, f in lst:
            all_feats.append(f)

    if not all_feats:
        return features_dict, None

    # Determina lunghezza massima e crea matrice con padding a zero
    max_len = max(f.shape[0] for f in all_feats)
    stacked = np.zeros((len(all_feats), max_len), dtype=np.float32)
    for i, f in enumerate(all_feats):
        stacked[i, : f.shape[0]] = f

    # Se pochi campioni, non scalare (evita crash su single-sample)
    if stacked.shape[0] < 2:
        return features_dict, None

    # Fit dello scaler sui vettori padded
    scaler = StandardScaler().fit(stacked)

    # Trasforma e rimappa al formato originale (tagliando il padding)
    idx = 0
    for cls, img_feats_list in features_dict.items():
        for i, (path, feat) in enumerate(img_feats_list):
            padded = np.zeros((max_len,), dtype=np.float32)
            padded[: feat.shape[0]] = feat
            scaled = scaler.transform(padded.reshape(1, -1)).flatten()
            # Riporta alla dimensione originale (rimuovendo eventuale padding)
            features_dict[cls][i] = (path, scaled[: feat.shape[0]])
            idx += 1

    return features_dict, scaler
