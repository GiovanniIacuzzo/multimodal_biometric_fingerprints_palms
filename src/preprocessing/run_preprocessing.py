"""
run_preprocessing.py (versione robusta e tracciabile)
---------------------------------------------------
Esegue il preprocessing su tutte le immagini del dataset
usando la pipeline di enhancement_advanced.py con:
- logging dettagliato
- controllo path
- gestione errori migliorata
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from src.preprocessing.enhancement import preprocess_fingerprint

# ==========================
# CONFIG
# ==========================

CATALOG_PATH = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/metadata/catalog.csv"
OUTPUT_DIR = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/processed"
LOG_PATH = os.path.join(OUTPUT_DIR, "errors_log.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_WORKERS = min(8, os.cpu_count())

# ==========================
# FUNZIONE DI PROCESSAMENTO
# ==========================

def process_single_image(row):
    img_path = row["path"]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # --- Controllo path ---
        if not os.path.exists(img_path):
            return {"status": "MISSING", "path": img_path, "error": "File non trovato"}

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"status": "UNREADABLE", "path": img_path, "error": "cv2.imread() ha restituito None"}

        # --- Preprocessing ---
        results = preprocess_fingerprint(img, debug_dir=save_dir)

        # --- Salvataggio risultati ---
        for key, img_res in results.items():
            if img_res is not None and img_res.size > 0:
                cv2.imwrite(os.path.join(save_dir, f"{key}.png"), img_res)

        return {"status": "OK", "path": img_path, "error": None}

    except Exception as e:
        err_trace = traceback.format_exc(limit=2)
        return {"status": "ERROR", "path": img_path, "error": f"{type(e).__name__}: {e}\n{err_trace}"}

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

    # --- Analisi risultati ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(LOG_PATH, index=False)

    ok_count = (df_results["status"] == "OK").sum()
    miss_count = (df_results["status"] == "MISSING").sum()
    unread_count = (df_results["status"] == "UNREADABLE").sum()
    err_count = (df_results["status"] == "ERROR").sum()

    print(f"\n✅ Preprocessing completato! File salvati in: {OUTPUT_DIR}")
    print(f"   ✅ OK: {ok_count} | ⚠️ Missing: {miss_count} | 🟥 Unreadable: {unread_count} | ❌ Error: {err_count}")
    print(f"   🔍 Log dettagliato: {LOG_PATH}")

if __name__ == "__main__":
    main()
