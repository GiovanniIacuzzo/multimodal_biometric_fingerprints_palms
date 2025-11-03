"""
run_preprocessing.py (versione robusta e parallela)
---------------------------------------------------
Esegue il preprocessing su tutte le immagini del dataset
usando la pipeline di enhancement.py con parallelizzazione
e gestione robusta degli errori.
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.preprocessing.enhancement import preprocess_fingerprint

# ==========================
# CONFIG
# ==========================

CATALOG_PATH = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/metadata/catalog.csv"
OUTPUT_DIR = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = min(8, os.cpu_count())  # numero di thread/processi paralleli


# ==========================
# FUNZIONE DI PROCESSAMENTO
# ==========================

def process_single_image(row):
    base_name = os.path.splitext(os.path.basename(row["path"]))[0]
    save_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"[SKIP] {row['path']} non leggibile"

        results = preprocess_fingerprint(img)

        # Salvataggio risultati intermedi
        for key, img_res in results.items():
            save_path = os.path.join(save_dir, f"{key}.png")
            cv2.imwrite(save_path, img_res)

        return f"[OK] {row['path']} processata"

    except Exception as e:
        return f"[ERROR] {row['path']} -> {e}"


# ==========================
# MAIN LOOP PARALLELIZZATO
# ==========================

def main():
    df = pd.read_csv(CATALOG_PATH)
    print(f"📁 Totale immagini da processare: {len(df)}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_image, row): row for _, row in df.iterrows()}

        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            results.append(f.result())

    # Log finale
    ok_count = sum(1 for r in results if r.startswith("[OK]"))
    skip_count = sum(1 for r in results if r.startswith("[SKIP]"))
    err_count = sum(1 for r in results if r.startswith("[ERROR]"))

    print(f"\n✅ Preprocessing completato! File salvati in: {OUTPUT_DIR}")
    print(f"   ✅ OK: {ok_count} | ⚠️ SKIP: {skip_count} | ❌ ERROR: {err_count}")


if __name__ == "__main__":
    main()
