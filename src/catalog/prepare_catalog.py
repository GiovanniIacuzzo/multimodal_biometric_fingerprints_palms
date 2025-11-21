import os
import re
import cv2
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style

from config import config_fingerprint

# ==========================================================
# REGEX SUPPORTATE
# ==========================================================
PATTERN_STANDARD = re.compile(r"(\d+)_(\d+)_(\d+)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)

PATTERN_NIST = re.compile(r"F(\d{4})_(\d+)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
PATTERN_S = re.compile(r"S(\d{4})_(\d+)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ==========================================================
# PARSER FILENAME
# ==========================================================
def parse_filename(filename: str):
    """
    Riconosce pattern:
      1. 3_1_1.jpg  (standard)
      2. F0003_10.bmp (NIST-like)
      3. S1387_02.bmp (nuovo formato)
    Restituisce: subject_id, finger_id, session_id
    """

    filename = filename.strip()

    m = PATTERN_STANDARD.search(filename)
    if m:
        subject_id, finger_id, session_id, _ = m.groups()
        return int(subject_id), int(finger_id), int(session_id)

    m = PATTERN_NIST.search(filename)
    if m:
        subject_raw, finger_id, _ = m.groups()
        subject_id = int(subject_raw)
        finger_id = int(finger_id)
        session_id = 1  # default per NIST
        return subject_id, finger_id, session_id

    m = PATTERN_S.search(filename)
    if m:
        subject_raw, finger_id, _ = m.groups()
        subject_id = int(subject_raw)
        finger_id = int(finger_id)
        session_id = 1  # default
        return subject_id, finger_id, session_id
    return None, None, None

def console_step(title):
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


# ==========================================================
# SCANSIONE CLUSTER
# ==========================================================
def scan_cluster(cluster_path: str, cluster_name: str):
    records = []
    console_step(f"Scansione cluster: {cluster_name}")

    files = [
        f for f in os.listdir(cluster_path)
        if f.lower().endswith(VALID_EXT)
    ]

    print(f"→ Immagini trovate: {len(files)}")

    for file in tqdm(files, desc=f"{Fore.MAGENTA}Processing{Style.RESET_ALL}", ncols=90):
        file_path = os.path.join(cluster_path, file)
        subject_id, finger_id, session_id = parse_filename(file)

        if None in (subject_id, finger_id, session_id):
            print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Filename non riconosciuto: {file}")
            continue

        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Immagine non leggibile")
            height, width = img.shape
        except Exception as e:
            print(f"{Fore.RED}[ERRORE]{Style.RESET_ALL} Lettura fallita {file_path}: {e}")
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

    print(f"{Fore.GREEN}✔ Cluster {cluster_name} completato — immagini valide: {len(records)}{Style.RESET_ALL}")
    return records


# ==========================================================
# SCANSIONE DATASET
# ==========================================================
def scan_dataset(dataset_root: str):
    console_step("Scansione dataset")
    print(f"→ Percorso dataset: {os.path.abspath(dataset_root)}")

    records = []
    clusters = [
        entry for entry in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, entry)) and entry.lower().startswith("cluster_")
    ]

    if not clusters:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Nessun cluster trovato.")
        return records

    for cluster_name in sorted(clusters):
        cluster_path = os.path.join(dataset_root, cluster_name)
        records.extend(scan_cluster(cluster_path, cluster_name))

    print(f"{Fore.GREEN}✔ Scansione completa — immagini totali: {len(records)}{Style.RESET_ALL}")
    return records


# ==========================================================
# SALVATAGGIO CSV
# ==========================================================
def save_catalog(records, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(records)

    if df.empty:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Nessuna immagine trovata.")
        return

    df.sort_values(by=["cluster_name", "subject_id", "finger_id", "session_id"],
                   inplace=True, ignore_index=True)

    df.to_csv(output_csv, index=False)

    console_step("Salvataggio catalogo")
    print(f"{Fore.GREEN}✔ File salvato:{Style.RESET_ALL} {output_csv}")
    print(f"→ Totale immagini: {len(df)}")
    print(f"→ Cluster unici: {df['cluster_name'].nunique()}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    console_step("Pipeline Scan Dataset")

    dataset_root = os.path.join(config_fingerprint.DATASET_DIR, "sorted_dataset")
    output_csv = os.path.join(config_fingerprint.METADATA_DIR, "catalog.csv")

    if not os.path.exists(dataset_root):
        print(f"{Fore.RED}[ERRORE]{Style.RESET_ALL} Percorso inesistente: {dataset_root}")
        return

    records = scan_dataset(dataset_root)
    if not records:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Nessuna immagine trovata.")
        return

    save_catalog(records, output_csv)
    print(f"{Fore.CYAN}✨ Pipeline completata con successo! ✨{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
