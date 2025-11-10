import os
import re
import cv2
import pandas as pd
from config import config_fingerprint

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
                    if img is None:
                        raise ValueError("Immagine non leggibile o corrotta.")
                    height, width = img.shape
                except Exception as e:
                    print(f"[ERRORE] Impossibile leggere {file_path}: {e}")
                    continue

                image_id = os.path.splitext(file)[0]
                records.append({
                    "image_id": image_id,
                    "subject_id": subject_id,
                    "finger_id": finger_id,
                    "session_id": session_id,
                    "path": os.path.abspath(file_path),
                    "width": width,
                    "height": height,
                    "format": "jpg",
                })

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

    dataset_root = config_fingerprint.DATASET_DIR
    output_csv = os.path.join(config_fingerprint.METADATA_DIR, "catalog.csv")

    records = scan_dataset(dataset_root)
    if not records:
        print("Nessuna immagine trovata. Controlla il percorso del dataset.")
        return

    save_catalog(records, output_csv)


if __name__ == "__main__":
    main()
