import cv2
import numpy as np
from pathlib import Path


def resize_and_normalize(img, size=(256, 256)):
    """Resize e normalizzazione in [0,1]."""
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    img_norm = img_resized / 255.0
    return img_norm


def local_contrast_normalization(img, kernel_size=15):
    """Normalizzazione locale robusta (illumination normalization)."""
    mean_local = cv2.blur(img, (kernel_size, kernel_size))
    std_local = cv2.blur((img - mean_local) ** 2, (kernel_size, kernel_size)) ** 0.5
    std_local = np.clip(std_local, 1e-6, None)
    img_norm = (img - mean_local) / std_local
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    return img_norm


def estimate_dominant_orientation(img):
    """Stima della direzione predominante tramite gradienti."""
    gy, gx = np.gradient(img)
    orientation = np.arctan2(gy, gx)
    hist, bins = np.histogram(orientation, bins=180, range=(-np.pi, np.pi))
    dominant_angle = bins[np.argmax(hist)]
    return dominant_angle


def align_image(img, angle_rad):
    """Allinea l'immagine secondo l'angolo dominante."""
    angle_deg = np.degrees(angle_rad)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned


def preprocess_image(img_path_or_array, resize=(256, 256), local_norm=True, align=True):
    """Pipeline completa di preprocessing per fingerprint."""
    if isinstance(img_path_or_array, (str, Path)):
        img = cv2.imread(str(img_path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(resize, dtype=np.uint8)
    else:
        img = img_path_or_array

    img = resize_and_normalize(img, size=resize)

    if local_norm:
        img = local_contrast_normalization(img)

    if align:
        try:
            angle = estimate_dominant_orientation(img)
            img = align_image(img, angle)
        except Exception:
            pass  # in caso di immagini vuote o piatte

    return img
