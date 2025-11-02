# ============================
# run_pipeline.py
# ============================
import sys
import os
import traceback

# Aggiunge la root del progetto ai path importabili
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa i moduli della pipeline
from scripts.config import ensure_dirs
from scripts.config import PROCESSED_DIR, FINGERPRINT_MODEL
from src.data.prepare_catalog import main as prepare_catalog
from src.preprocessing.run_preprocessing import main as run_preprocessing
from src.features.minutiae_extraction import main as extract_minutiae
from src.features.descriptors_handcrafted import main as extract_handcrafted
from src.models.descriptors_deep import main as train_deep_features
from src.matching.run_matching import main as run_matching
from src.evaluation.evaluate_results import main as evaluate

print("====================================")
print("🧬 MULTIMODAL BIOMETRIC PIPELINE")
print("====================================")

def run_pipeline():
    ensure_dirs()

    try:
        """ print("\n[1/6] 🔍 Preparazione catalogo...")
        prepare_catalog()

        print("\n[2/6] 🧼 Preprocessing immagini...")
        run_preprocessing()

        print("\n[3/6] 🔩 Estrazione minutiae...")
        extract_minutiae()

        print("\n[4/6] 🔬 Estrazione descrittori (handcrafted + deep)...")
        extract_handcrafted() """

        # Training o estrazione feature deep
        print("   ↳ Avvio training modello deep CNN...")
        train_deep_features(
            dataset_dir=PROCESSED_DIR,
            save_path=FINGERPRINT_MODEL,
            epochs=5,
            batch_size=16,
            embedding_dim=256,
            lr=1e-4,
            device=None
        )

        print("\n[5/6] 🤝 Matching e fusione punteggi...")
        run_matching()

        print("\n[6/6] 📊 Valutazione risultati...")
        evaluate()

        print("\n✅ Pipeline completata con successo!")

    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione della pipeline:\n{e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
