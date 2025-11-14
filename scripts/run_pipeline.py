import sys
import os
import traceback
import time
import json
import faulthandler
faulthandler.enable()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_fingerprint import get_path
from src.catalog.prepare_catalog import main as prepare_catalog
from src.preprocessing.run_preprocessing import run_preprocessing
from src.features.extract_features import main as extract_minutiae
from src.matching.match_features import batch_match_from_debug
from src.db.database import clear_database, get_all_image_filenames
from src.evaluation.evaluate_performance import evaluate_results


# ======================================
# Pipeline principale
# ======================================
def run_pipeline():
    # Crea directory necessarie
    paths_to_create = ["processed_dir", "features_dir", "metadata_dir"]

    for key in paths_to_create:
        path = get_path(key, "./" + key)  # legge da YAML o usa il default
        os.makedirs(path, exist_ok=True)

    try:
        print("====================================")
        print("  MULTIMODAL BIOMETRIC PIPELINE")
        print("====================================")
        clear_database()

        # -------------------------------------------------
        print("\n[1/5] Preparazione catalogo...")
        prepare_catalog()

        # -------------------------------------------------
        print("\n[2/5] Preprocessing immagini...")
        start_pre = time.time()
        run_preprocessing(
            input_dir=get_path("dataset_dir", "./dataset/sorted_dataset"),
            output_dir=get_path("processed_dir", "./dataset/processed"),
            debug=True,
            small_subset=False
        )
        print(f"[INFO] Preprocessing completato in {time.time()-start_pre:.2f} sec")

        """ # -------------------------------------------------
        print("\n[3/5] Estrazione minutiae...")
        start_feat = time.time()
        extract_minutiae()
        print(f"[INFO] Estrazione minutiae completata in {time.time()-start_feat:.2f} sec")

        # -------------------------------------------------
        print("\n[4/5] Matching impronte...")
        filenames = get_all_image_filenames()
        match_results = batch_match_from_debug(debug_dir="data/processed/debug", device="mps")

        print("\nRisultati matching:")
        for pair, score in match_results.items():
            print(f"{pair[0]} vs {pair[1]} -> Similarità: {score:.2f}")
        
        results_path = os.path.join(config_fingerprint.METADATA_DIR, "match_results.json")

        json_results = {f"{str(k[0])}_vs_{str(k[1])}": v for k, v in match_results.items()}

        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=4)

        # -------------------------------------------------
        print("\n[5/5] Valutazione performance...")

        # Valutazione finale
        metrics = evaluate_results(results_path, output_path=os.path.join(config_fingerprint.METADATA_DIR, "performance_metrics.json"), plot_dir=config_fingerprint.METADATA_DIR)
 """
        print("\nPipeline completata con successo!")

    except Exception as e:
        print(f"\nErrore durante l'esecuzione della pipeline:\n{e}")
        traceback.print_exc()

# ======================================
# Entry point
# ======================================
if __name__ == "__main__":
    run_pipeline()
