import os
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
CONFIG_YAML = os.path.join(BASE_DIR, "config", "config_fingerprint.yml")

with open(CONFIG_YAML, "r") as f:
    cfg = yaml.safe_load(f)

# Utility per ottenere percorso assoluto
def get_path(key: str, default: str) -> str:
    path = cfg.get("paths", {}).get(key, default)
    return os.path.abspath(os.path.join(BASE_DIR, path))

METADATA_DIR   = get_path("metadata_dir", "data/metadata")
DATASET_DIR   = get_path("dataset_dir", "./dataset")
SORTED_DATASET_DIR = get_path("sorted_dataset_dir", "./dataset/sorted_dataset")
PROCESSED_DIR = get_path("processed_dir", "./dataset/processed")
FEATURES_DIR  = get_path("features_dir", "./data/features")
DEBUG_DIR     = get_path("debug_dir", "./data/features/debug")

DB_CONFIG = cfg.get("database", {})

# Parametri
PREPROCESSING_PARAMS = cfg.get("preprocessing", {})
BINARIZATION_PARAMS = cfg.get("binarization", {})
ORIENTATION_PARAMS = cfg.get("orientation", {})
GENERAL_PARAMS = cfg.get("general", {})

# Stampa riassuntiva
def print_config_summary():
    print("\n=== CONFIGURAZIONE CARICATA ===")
    print("Percorsi:")
    for k, v in cfg.get("paths", {}).items():
        print(f"  {k}: {get_path(k, v)}")
    print("\nDatabase:")
    for k, v in DB_CONFIG.items():
        print(f"  {k}: {v}")
    print("\nParametri preprocessing:", PREPROCESSING_PARAMS)
    print("Parametri binarizzazione:", BINARIZATION_PARAMS)
    print("Parametri orientazione:", ORIENTATION_PARAMS)
    print("Parametri generali:", GENERAL_PARAMS)
    print("================================\n")
