import cv2
import numpy as np
from pathlib import Path

# ==============================================================
# Resize e conversione in float [0,1]
# ==============================================================
def resize_and_normalize(img, size=(256, 256)):
    """
    Resize e normalizzazione globale [0,255] -> [0,1]
    """
    img_resized = cv2.resize(img, size).astype(np.float32)
    img_norm = img_resized / 255.0
    return img_norm


# ==============================================================
# Normalizzazione locale (illumination normalization)
# ==============================================================
def local_contrast_normalization(img, kernel_size=15):
    """
    Applica normalizzazione locale:
    (img - mean_local) / (std_local + eps)
    img: numpy array HxW float [0,1]
    """
    mean_local = cv2.blur(img, (kernel_size, kernel_size))
    std_local = cv2.blur((img - mean_local)**2, (kernel_size, kernel_size))**0.5 + 1e-8
    img_norm = (img - mean_local) / std_local
    # Rimappo in [0,1]
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    return img_norm


# ==============================================================
# Stima orientazione principale (soft alignment)
# ==============================================================
def estimate_dominant_orientation(img):
    """
    Calcola la direzione predominante dell’immagine tramite gradienti.
    Ritorna l’angolo in radianti
    """
    gy, gx = np.gradient(img)
    orientation = np.arctan2(gy, gx)
    hist, bins = np.histogram(orientation, bins=180, range=(-np.pi, np.pi))
    dominant_angle = bins[np.argmax(hist)]
    return dominant_angle


def align_image(img, angle_rad):
    """
    Allinea l’immagine secondo l’angolo stimato (in radianti)
    """
    angle_deg = np.degrees(angle_rad)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return aligned


# ==============================================================
# Pipeline completa di preprocessing
# ==============================================================
def preprocess_image(img_path_or_array, resize=(256, 256), local_norm=True, align=True):
    # Se è un path, carica l'immagine
    if isinstance(img_path_or_array, (str, Path)):
        img = cv2.imread(str(img_path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # fallback: immagine nera
            img = np.zeros(resize, dtype=np.uint8)
    else:
        img = img_path_or_array

    img = resize_and_normalize(img, size=resize)

    if local_norm:
        img = local_contrast_normalization(img)

    if align:
        angle = estimate_dominant_orientation(img)
        img = align_image(img, angle)

    return img
