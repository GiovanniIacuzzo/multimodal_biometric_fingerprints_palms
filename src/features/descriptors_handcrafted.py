"""
descriptors_handcrafted.py
--------------------------
Calcolo dei descrittori LBP e Gabor attorno alle minutiae per fingerprint feature extraction.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from scripts.config import PROCESSED_DIR, FEATURES_DIR, DATASET_DIR

# ==========================
# PARAMETRI GABOR / LBP
# ==========================
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
GABOR_FREQUENCIES = [0.1, 0.2, 0.3]
GABOR_THETAS = np.linspace(0, np.pi, 8, endpoint=False)
PATCH_SIZE = 32

# ==========================
# FUNZIONI BASE
# ==========================

def extract_patch(image: np.ndarray, x: int, y: int, size: int = PATCH_SIZE) -> np.ndarray:
    half = size // 2
    h, w = image.shape
    x_min, x_max = max(0, x - half), min(w, x + half)
    y_min, y_max = max(0, y - half), min(h, y + half)
    patch = image[y_min:y_max, x_min:x_max]
    return cv2.copyMakeBorder(
        patch,
        top=max(0, half - y_min),
        bottom=max(0, (y + half) - h),
        left=max(0, half - x_min),
        right=max(0, (x + half) - w),
        borderType=cv2.BORDER_REFLECT
    )

def compute_lbp_descriptor(patch: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(patch, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def compute_gabor_descriptor(patch: np.ndarray) -> np.ndarray:
    responses = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_THETAS:
            real, imag = gabor(patch, frequency=freq, theta=theta)
            mag = np.sqrt(real**2 + imag**2)
            responses.append([mag.mean(), mag.std()])
    return np.array(responses).flatten()

def extract_features_for_minutia(image: np.ndarray, x: int, y: int) -> np.ndarray:
    patch = extract_patch(image, x, y)
    lbp_feat = compute_lbp_descriptor(patch)
    gabor_feat = compute_gabor_descriptor(patch)
    return np.concatenate([lbp_feat, gabor_feat])

def process_image(image_path: str, minutiae_path: str, output_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    with open(minutiae_path) as f:
        minutiae = json.load(f)

    features = []
    for m in minutiae:
        vec = extract_features_for_minutia(image, m["x"], m["y"])
        features.append(vec)

    features = np.array(features)
    np.save(output_path, features)
    return features

# ==========================
# MAIN
# ==========================

def main():
    """Esegue la generazione delle feature LBP + Gabor per tutte le impronte."""
    catalog_path = os.path.join(DATASET_DIR, "catalog.csv")
    minutiae_dir = os.path.join(FEATURES_DIR, "minutiae")
    output_dir = os.path.join(FEATURES_DIR, "descriptors_handcrafted")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(catalog_path)

    print(f"🧠 Estrazione feature handcrafted da {len(df)} impronte...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_name = os.path.splitext(os.path.basename(row["path"]))[0]
        image_path = os.path.join(PROCESSED_DIR, base_name, "enhanced.png")
        minutiae_path = os.path.join(minutiae_dir, f"{base_name}_minutiae.json")
        output_path = os.path.join(output_dir, f"{base_name}_features.npy")

        if os.path.exists(image_path) and os.path.exists(minutiae_path):
            process_image(image_path, minutiae_path, output_path)

    print(f"\n✅ Feature extraction completata! File salvati in: {output_dir}")

if __name__ == "__main__":
    main()
