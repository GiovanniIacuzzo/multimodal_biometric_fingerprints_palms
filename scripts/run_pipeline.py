import sys
import os
import traceback
import faulthandler

faulthandler.enable()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa i moduli della pipeline
from config import config
from src.catalog.prepare_catalog import main as prepare_catalog
from src.preprocessing.run_preprocessing import run_preprocessing
from src.features.extract_features import main as extract_minutiae


def run_pipeline():
    # Assicurati che le directory esistano
    for path in [
        config.DATA_DIR,
        config.PROCESSED_DIR,
        config.FEATURES_DIR,
        config.METADATA_DIR,
        config.LOG_DIR,
        config.TEMP_DIR
    ]:
        os.makedirs(path, exist_ok=True)

    try:
        print("====================================")
        print("  MULTIMODAL BIOMETRIC PIPELINE")
        print("====================================")

        print("\n[1/6] Preparazione catalogo...")
        prepare_catalog()  # userà internamente config.METADATA_DIR

        print("\n[2/6] Preprocessing immagini...")
        run_preprocessing(
            input_dir=config.DATASET_DIR,
            output_dir=config.PROCESSED_DIR,
            debug=True,
            small_subset=True
        )

        print("\n[3/6] Estrazione minutiae...")
        extract_minutiae()  # anche qui potrà leggere da config.FEATURES_DIR

        print("\n✅ Pipeline completata con successo!")

    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione della pipeline:\n{e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_pipeline()
