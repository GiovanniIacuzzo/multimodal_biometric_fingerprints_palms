"""
prepare_catalog.py
------------------
Crea il catalogo del dataset PolyU HRF DBII scansionando la directory delle immagini .jpg
e generando un file CSV con i metadati necessari.

Output:
    data/metadata/catalog.csv
"""

import os
import re
import cv2
import pandas as pd

DATASET_DIR = os.path.abspath("dataset/DBII/metadata")

# ==============================
# CONFIGURAZIONE
# ==============================

DATASET_ROOT = "dataset/DBII"
OUTPUT_CSV = os.path.join(DATASET_DIR, "catalog.csv")

# Regex per parsing del nome file: es. 001_2_1.jpg
FILENAME_PATTERN = re.compile(r"(\d+)_(\d+)_(\d+)\.jpg$", re.IGNORECASE)


def parse_filename(filename: str):
    """
    Estrae (subject_id, finger_id, session_id) dal nome del file.
    """
    match = FILENAME_PATTERN.search(filename)
    if match:
        subject_id, finger_id, session_id = match.groups()
        return subject_id, finger_id, session_id
    return None, None, None


def scan_dataset(dataset_root: str):
    """
    Scansiona ricorsivamente il dataset per trovare tutte le immagini .jpg
    e raccoglie i metadati.
    """
    records = []

    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(".jpg"):
                file_path = os.path.join(root, file)
                subject_id, finger_id, session_id = parse_filename(file)

                # Carica l'immagine per ottenere dimensioni
                try:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    height, width = img.shape
                except Exception as e:
                    print(f"[ERRORE] Impossibile leggere {file_path}: {e}")
                    continue

                image_id = os.path.splitext(file)[0]
                record = {
                    "image_id": image_id,
                    "subject_id": subject_id,
                    "finger_id": finger_id,
                    "session_id": session_id,
                    "path": os.path.abspath(file_path),
                    "width": width,
                    "height": height,
                    "format": "jpg",
                }
                records.append(record)

    return records


def save_catalog(records, output_csv):
    """
    Salva i record in un file CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nCatalogo salvato in: {output_csv}")
    print(f"Totale immagini: {len(df)}")


def main():
    print("Scansione del dataset PolyU HRF DBII...")
    records = scan_dataset(DATASET_ROOT)
    if not records:
        print("Nessuna immagine trovata. Controlla il percorso del dataset.")
        return
    save_catalog(records, OUTPUT_CSV)


if __name__ == "__main__":
    main()
