import sys
import os
import traceback
import time
import faulthandler
faulthandler.enable()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from src.catalog.prepare_catalog import main as prepare_catalog
from src.preprocessing.run_preprocessing import run_preprocessing
from src.features.extract_features import main as extract_minutiae
from src.matching.match_features import batch_match
from src.db.database import get_connection, clear_database, get_all_image_ids

# ======================================
# Pipeline principale
# ======================================
def run_pipeline():
    # Crea directory necessarie
    for path in [config.DATA_DIR, config.PROCESSED_DIR, config.FEATURES_DIR, config.METADATA_DIR]:
        os.makedirs(path, exist_ok=True)

    try:
        print("====================================")
        print("  MULTIMODAL BIOMETRIC PIPELINE")
        print("====================================")
        clear_database()

        # -------------------------------------------------
        print("\n[1/4] Preparazione catalogo...")
        prepare_catalog()

        # -------------------------------------------------
        print("\n[2/4] Preprocessing immagini...")
        start_pre = time.time()
        run_preprocessing(
            input_dir=config.DATASET_DIR,
            output_dir=config.PROCESSED_DIR,
            debug=True,
            small_subset=True
        )
        print(f"[INFO] Preprocessing completato in {time.time()-start_pre:.2f} sec")

        # -------------------------------------------------
        print("\n[3/4] Estrazione minutiae...")
        start_feat = time.time()
        extract_minutiae()
        print(f"[INFO] Estrazione minutiae completata in {time.time()-start_feat:.2f} sec")

        # -------------------------------------------------
        print("\n[4/4] Matching impronte...")
        image_ids = get_all_image_ids()
        match_results = batch_match(image_ids)

        print("\nRisultati matching:")
        for pair, score in match_results.items():
            print(f"{pair[0]} vs {pair[1]} -> Similarità: {score:.2f}")

        print("\nPipeline completata con successo!")

    except Exception as e:
        print(f"\nErrore durante l'esecuzione della pipeline:\n{e}")
        traceback.print_exc()

# ======================================
# Entry point
# ======================================
if __name__ == "__main__":
    run_pipeline()
