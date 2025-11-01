from pathlib import Path
import os
from dotenv import load_dotenv

# Carica variabili da .env
load_dotenv(dotenv_path=Path(__file__).parent / "config.env")

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
DATASET_DIR = Path(os.getenv("DATASET_DIR", "../dataset/DBII"))
OUTPUT_CSV = Path(os.getenv("OUTPUT_CSV", "labels.csv"))
FIGURE_DIR = Path(os.getenv("IMG", "../results/img"))

# Lettura scale multiple
IMG_SCALES = [
    (int(os.getenv("IMG_SCALE_1", 128)), int(os.getenv("IMG_SCALE_1", 128))),
    (int(os.getenv("IMG_SCALE_2", 64)), int(os.getenv("IMG_SCALE_2", 64))),
    (int(os.getenv("IMG_SCALE_3", 32)), int(os.getenv("IMG_SCALE_3", 32)))
]

BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", 32))
GAUSSIAN_BLUR = int(os.getenv("GAUSSIAN_BLUR", 5))

USE_HOG = os.getenv("USE_HOG", "True") == "True"
USE_GABOR = os.getenv("USE_GABOR", "True") == "True"
USE_LBP = os.getenv("USE_LBP", "True") == "True"

MAX_K_CLUSTERS = int(os.getenv("MAX_K_CLUSTERS", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 5))
MAX_SAMPLES_PER_CLASS = int(os.getenv("MAX_SAMPLES_PER_CLASS", 200))
