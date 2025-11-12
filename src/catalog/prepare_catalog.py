import os
import re
import cv2
import pandas as pd
from tqdm import tqdm
from config import config_fingerprint

FILENAME_PATTERN = re.compile(r"(\d+)_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_filename(filename: str):
    """
    Estrae (subject_id, finger_id, session_id) dal nome del file.
    """
    match = FILENAME_PATTERN.search(filename)
    if match:
        subject_id, finger_id, session_id, _ = match.groups()
        return subject_id, finger_id, session_id
    return None, None, None

def infer_cluster_from_path(root_path: str, dataset_root: str):
    """
    Estrae il nome del cluster dal percorso.
    Esempio: ./dataset/cluster_0/subject1/... -> cluster_0
    """
    rel_path = os.path.relpath(root_path, dataset_root)
    parts = rel_path.split(os.sep)
    for p in parts:
        if p.lower().startswith("cluster_"):
            return p
    return None


def scan_dataset(dataset_root: str):
    """
    Scansiona ricorsivamente il dataset per trovare tutte le immagini
    e raccoglie metadati, incluso il cluster di appartenenza.
    """
    records = []

    print(f"\n[INFO] Scansione dataset in: {os.path.abspath(dataset_root)}\n")

    for root, _, files in os.walk(dataset_root):
        cluster_name = infer_cluster_from_path(root, dataset_root)

        for file in tqdm(files, desc=f"Scanning {cluster_name or 'no_cluster'}", leave=False):
            if not re.search(r"\.(jpg|jpeg|png)$", file, re.IGNORECASE):
                continue

            file_path = os.path.join(root, file)
            subject_id, finger_id, session_id = parse_filename(file)

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
                "cluster_name": cluster_name,
                "path": os.path.abspath(file_path),
                "width": width,
                "height": height,
                "format": os.path.splitext(file)[1].lower().strip("."),
            })

    return records


def save_catalog(records, output_csv):
    """
    Salva i record in un file CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)

    # Ordina per cluster e subject_id
    df.sort_values(by=["cluster_name", "subject_id"], inplace=True, ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"\nCatalogo salvato in: {output_csv}")
    print(f"Totale immagini: {len(df)}")
    print(f"Cluster unici trovati: {df['cluster_name'].nunique()}")


def main():
    print("Scansione del dataset con struttura a cluster...\n")

    dataset_root = config_fingerprint.DATASET_DIR
    print(f"Percorso dataset: {dataset_root}")
    output_csv = os.path.join(config_fingerprint.METADATA_DIR, "catalog.csv")

    if not os.path.exists(dataset_root):
        print(f"[ERRORE] Il percorso del dataset non esiste: {dataset_root}")
        return

    records = scan_dataset(dataset_root)
    if not records:
        print("Nessuna immagine trovata. Controlla la struttura del dataset.")
        return

    save_catalog(records, output_csv)


if __name__ == "__main__":
    main()
