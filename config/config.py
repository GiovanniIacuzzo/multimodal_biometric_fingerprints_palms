# ============================
# config.py
# ============================
import os

# ----------------------------
# PATH PRINCIPALI
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATASET = os.path.abspath("dataset/DBII")
DATASET_DIR = os.path.join(BASE_DIR, "data", "metadata")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")

# ----------------------------
# FILE SPECIFICI
# ----------------------------
CATALOG_CSV = os.path.join(DATASET_DIR, "catalog.csv")
FINGERPRINT_MODEL = os.path.join(MODELS_DIR, "fingerprint_deep.pt")

# ----------------------------
# PARAMETRI PIPELINE
# ----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
SEED = 42

# ----------------------------
# MODALITÀ DI ESECUZIONE
# ----------------------------
USE_GPU = True
SAVE_INTERMEDIATE = True
ENABLE_AUGMENTATION = True

# ----------------------------
# UTILS
# ----------------------------
def ensure_dirs():
    """Crea le directory principali se non esistono."""
    for d in [DATASET_DIR, PROCESSED_DIR, FEATURES_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
