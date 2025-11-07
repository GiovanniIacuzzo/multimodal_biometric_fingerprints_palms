import os
import yaml
from dotenv import load_dotenv

# ==============================
# LOAD ENVIRONMENT VARIABLES (.env)
# ==============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
ENV_PATH = os.path.join(CONFIG_DIR, ".env")
YAML_PATH = os.path.join(CONFIG_DIR, "config_path.yml")

# Carica variabili da .env
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print("[AVVISO] File .env non trovato, uso valori di default.")

# Carica percorsi da config_path.yml
if os.path.exists(YAML_PATH):
    with open(YAML_PATH, "r") as f:
        path_cfg = yaml.safe_load(f)
else:
    print("[AVVISO] File config_path.yml non trovato, uso percorsi di default.")
    path_cfg = {}

# =====================================================
# 2. Configurazione database
# =====================================================
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "dbname": os.getenv("PGDATABASE", "biometria"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
    "port": os.getenv("PGPORT", "5432"),
}

# ==============================
# PATH CONFIGURATION
# ==============================
def _get_path(key: str, default: str) -> str:
    """Restituisce un percorso assoluto da YAML o il default."""
    path = path_cfg.get(key, default)
    return os.path.abspath(os.path.join(BASE_DIR, path))


DATA_DIR = _get_path("data_dir", "./data")
DATASET_DIR = _get_path("dataset_dir", "./dataset/DBII")
PROCESSED_DIR = _get_path("processed_dir", "./data/processed")
FEATURES_DIR = _get_path("features_dir", "./data/features")
METADATA_DIR = _get_path("metadata_dir", "./data/metadata")
LOG_DIR = _get_path("log_dir", "./logs")
TEMP_DIR = _get_path("temp_dir", "./temp")
DEBUG_DIR = _get_path("debug_dir", "./data/features/debug")


# ==============================
# PARAMETRI TECNICI DAL .ENV
# ==============================
def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (ValueError, TypeError):
        print(f"[AVVISO] Variabile {name} non valida, uso default={default}")
        return default


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):
        print(f"[AVVISO] Variabile {name} non valida, uso default={default}")
        return default


# Pruning e orientamento
PRUNE_ITERS = _getenv_int("PRUNE_ITERS", 2)
PRUNE_AREA = _getenv_int("PRUNE_AREA", 2)
ORIENT_SIGMA = _getenv_float("ORIENT_SIGMA", 3.0)

# CLAHE
CLAHE_CLIP_LIMIT = _getenv_float("CLAHE_CLIP_LIMIT", 2.0)
CLAHE_TILE_SIZE = _getenv_int("CLAHE_TILE_SIZE", 8)

# Filtri
BILATERAL_D = _getenv_int("BILATERAL_D", 5)
BILATERAL_SIGMA_COLOR = _getenv_float("BILATERAL_SIGMA_COLOR", 50.0)
BILATERAL_SIGMA_SPACE = _getenv_float("BILATERAL_SIGMA_SPACE", 7.0)
GAUSSIAN_SIGMA = _getenv_float("GAUSSIAN_SIGMA", 0.7)

# Segmentazione
SAUVOLA_WIN = _getenv_int("SAUVOLA_WIN", 25)
SAUVOLA_K = _getenv_float("SAUVOLA_K", 0.2)
LOCAL_PATCH = _getenv_int("LOCAL_PATCH", 64)
MIN_OBJ_SIZE = _getenv_int("MIN_OBJ_SIZE", 30)
MAX_HOLE_SIZE = _getenv_int("MAX_HOLE_SIZE", 100)
MIN_SEGMENT_AREA = _getenv_int("MIN_SEGMENT_AREA", 5000)

# Blocchi e soglie
BLOCK_SIZE = _getenv_int("BLOCK_SIZE", 16)
ENERGY_THRESHOLD = _getenv_float("ENERGY_THRESHOLD", 1e-2)
REL_THRESHOLD = _getenv_float("REL_THRESHOLD", 0.2)
VIS_SCALE = _getenv_int("VIS_SCALE", 8)


# ==============================
# UTILITY: STAMPA CONFIGURAZIONE
# ==============================
def print_config_summary():
    print("\n=== CONFIGURAZIONE CORRENTE ===")
    print("Percorsi:")
    for k, v in path_cfg.items():
        print(f"  {k}: {os.path.abspath(v)}")

    print("\nParametri:")
    keys = [
        "PRUNE_ITERS", "PRUNE_AREA", "ORIENT_SIGMA",
        "CLAHE_CLIP_LIMIT", "CLAHE_TILE_SIZE",
        "BILATERAL_D", "BILATERAL_SIGMA_COLOR", "BILATERAL_SIGMA_SPACE", "GAUSSIAN_SIGMA",
        "SAUVOLA_WIN", "SAUVOLA_K", "LOCAL_PATCH", "MIN_OBJ_SIZE", "MAX_HOLE_SIZE", "MIN_SEGMENT_AREA",
        "BLOCK_SIZE", "ENERGY_THRESHOLD", "REL_THRESHOLD", "VIS_SCALE"
    ]
    for k in keys:
        print(f"  {k}: {globals()[k]}")
    print("================================\n")
