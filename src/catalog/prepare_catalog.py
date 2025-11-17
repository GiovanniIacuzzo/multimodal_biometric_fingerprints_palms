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
        return int(subject_id), int(finger_id), int(session_id)
    return None, None, None


def scan_cluster(cluster_path: str, cluster_name: str):
    """
    Scansiona tutte le immagini in un singolo cluster.
    """
    records = []

    for root, _, files in os.walk(cluster_path):
        for file in files:
            if not re.search(r"\.(jpg|jpeg|png)$", file, re.IGNORECASE):
                continue

            file_path = os.path.join(root, file)
            subject_id, finger_id, session_id = parse_filename(file)

            if None in (subject_id, finger_id, session_id):
                print(f"[WARN] Filename non riconosciuto: {file}")
                continue

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


def scan_dataset(dataset_root: str):
    """
    Scansiona solo le cartelle cluster_* direttamente sotto dataset_root.
    """
    records = []

    print(f"\n[INFO] Scansione dataset in: {os.path.abspath(dataset_root)}\n")

    # Lista solo le cartelle cluster_*
    for entry in os.listdir(dataset_root):
        cluster_path = os.path.join(dataset_root, entry)
        if os.path.isdir(cluster_path) and entry.lower().startswith("cluster_"):
            print(f"[INFO] Scansione cluster: {entry}")
            records.extend(scan_cluster(cluster_path, entry))

    return records


def save_catalog(records, output_csv):
    """
    Salva i record in un file CSV ordinato.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)

    if df.empty:
        print("[WARN] Nessuna immagine trovata!")
        return

    # Ordina per cluster, subject, finger, session
    df.sort_values(by=["cluster_name", "subject_id", "finger_id", "session_id"],
                   inplace=True, ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"\nCatalogo salvato in: {output_csv}")
    print(f"Totale immagini: {len(df)}")
    print(f"Cluster unici trovati: {df['cluster_name'].nunique()}")


def main():
    print("Scansione del dataset (solo cluster_*):\n")

    dataset_root = config_fingerprint.DATASET_DIR + "/sorted_dataset"
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
