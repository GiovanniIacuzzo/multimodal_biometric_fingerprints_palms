"""
orientation_field.py
--------------------
Calcolo del campo di orientazione e frequenza locale
per immagini di impronte digitali (PolyU HRF DBII).

INPUT:  Immagine migliorata (.png) ottenuta dal preprocessing
OUTPUT: Mappe di orientazione e frequenza locali (.npy)

Autore: Giovanni [Cognome]
Corso: Sistemi di Identificazione Biometrica - Laurea Magistrale
Anno: 2025
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ==========================
# 1. ORIENTATION MAP
# ==========================

def compute_orientation_map(img: np.ndarray, block_size: int = 16, smooth: int = 3) -> np.ndarray:
    """
    Calcola la mappa di orientazione (in radianti) utilizzando
    gradienti locali su blocchi di dimensione block_size.
    """
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    rows, cols = img.shape
    orientation_map = np.zeros((rows // block_size, cols // block_size))

    for i in range(0, rows - block_size, block_size):
        for j in range(0, cols - block_size, block_size):
            block_Gx = Gx[i:i+block_size, j:j+block_size]
            block_Gy = Gy[i:i+block_size, j:j+block_size]

            Vx = np.sum(2 * block_Gx * block_Gy)
            Vy = np.sum(block_Gx**2 - block_Gy**2)

            theta = 0.5 * np.arctan2(Vx, Vy)
            orientation_map[i // block_size, j // block_size] = theta

    # Smoothing per coerenza direzionale
    return gaussian_filter(orientation_map, sigma=smooth)


# ==========================
# 2. FREQUENCY ESTIMATION
# ==========================

def estimate_frequency(img: np.ndarray, orientation_map: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Stima la frequenza locale delle creste utilizzando FFT
    sul profilo proiettato perpendicolarmente all’orientazione media.
    """
    rows, cols = img.shape
    freq_map = np.zeros_like(orientation_map)

    for i in range(0, rows - block_size, block_size):
        for j in range(0, cols - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            theta = orientation_map[i // block_size, j // block_size]

            # Rotazione del blocco per allineare le creste
            M = cv2.getRotationMatrix2D((block_size/2, block_size/2),
                                        np.degrees(theta + np.pi/2), 1)
            rotated = cv2.warpAffine(block, M, (block_size, block_size),
                                     flags=cv2.INTER_LINEAR)

            # Profilo medio lungo le righe
            projection = np.mean(rotated, axis=0)
            projection -= np.mean(projection)

            # FFT per identificare la frequenza dominante
            spectrum = np.abs(np.fft.fft(projection))
            freqs = np.fft.fftfreq(len(projection))

            idx = np.argmax(spectrum[1:len(spectrum)//2]) + 1
            dominant_freq = abs(freqs[idx])

            if 0.01 < dominant_freq < 0.25:
                freq_map[i // block_size, j // block_size] = dominant_freq

    return gaussian_filter(freq_map, sigma=1)


# ==========================
# 3. PIPELINE COMPLETA
# ==========================

def compute_orientation_and_frequency(img: np.ndarray) -> dict:
    """
    Calcola entrambe le mappe:
      - orientazione locale (radiani)
      - frequenza locale (cicli/pixel)
    """
    orientation = compute_orientation_map(img)
    frequency = estimate_frequency(img, orientation)
    return {"orientation_map": orientation, "frequency_map": frequency}


# ==========================
# 4. MAIN (eseguibile)
# ==========================

if __name__ == "__main__":
    # Percorsi configurabili
    CATALOG_PATH = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/metadata/catalog.csv"
    PROCESSED_DIR = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/processed"
    OUTPUT_DIR = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/features/orientation"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CATALOG_PATH)

    print(f"📊 Calcolo mappe di orientazione e frequenza per {len(df)} immagini...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_name = os.path.splitext(os.path.basename(row["path"]))[0]
        img_path = os.path.join(PROCESSED_DIR, base_name, "enhanced.png")

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        results = compute_orientation_and_frequency(img)

        # Salvataggio output
        np.save(os.path.join(OUTPUT_DIR, f"{base_name}_orientation.npy"), results["orientation_map"])
        np.save(os.path.join(OUTPUT_DIR, f"{base_name}_frequency.npy"), results["frequency_map"])

    print(f"\n✅ Mappe salvate in: {OUTPUT_DIR}")
