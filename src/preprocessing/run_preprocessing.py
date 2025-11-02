"""
run_preprocessing.py
--------------------
Esegue il preprocessing su tutte le immagini del dataset
usando la pipeline di enhancement.py.
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
from src.preprocessing.enhancement import preprocess_fingerprint

# ==========================
# CONFIG
# ==========================

CATALOG_PATH = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/metadata/catalog.csv"
OUTPUT_DIR = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# MAIN LOOP
# ==========================

def main():
    df = pd.read_csv(CATALOG_PATH)
    print(f"📁 Totale immagini da processare: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        results = preprocess_fingerprint(img)

        # Salvataggio risultati
        base_name = os.path.splitext(os.path.basename(row["path"]))[0]
        save_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(os.path.join(save_dir, "normalized.png"), results["normalized"])
        cv2.imwrite(os.path.join(save_dir, "enhanced.png"), results["enhanced"])
        cv2.imwrite(os.path.join(save_dir, "roi_mask.png"), results["roi_mask"])
        cv2.imwrite(os.path.join(save_dir, "binary.png"), results["binary"])
        cv2.imwrite(os.path.join(save_dir, "skeleton.png"), results["skeleton"])

    print(f"\n✅ Preprocessing completato! Risultati salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
