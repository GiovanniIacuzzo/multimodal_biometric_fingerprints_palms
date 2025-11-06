import sys
import os
import traceback
import faulthandler
faulthandler.enable()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa i moduli della pipeline
from config.config import ensure_dirs
from src.catalog.prepare_catalog import main as prepare_catalog
from src.preprocessing.run_preprocessing import run_preprocessing
from src.features.minutiae_extraction import main as extract_minutiae
from config.config import DATASET, PROCESSED_DIR

def run_pipeline():
    ensure_dirs()

    try:
        print("\n[1/6] Preparazione catalogo...")
        prepare_catalog()

        print("\n[2/6] Preprocessing immagini...")
        run_preprocessing(input_dir=DATASET, output_dir=PROCESSED_DIR, debug=True, small_subset=True)

        print("\n[3/6] Estrazione minutiae...")
        # extract_minutiae(test_mode=True)

        print("\nPipeline completata con successo!")

    except Exception as e:
        print(f"\nErrore durante l'esecuzione della pipeline:\n{e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("====================================")
    print("  MULTIMODAL BIOMETRIC PIPELINE")
    print("====================================")
    run_pipeline()
