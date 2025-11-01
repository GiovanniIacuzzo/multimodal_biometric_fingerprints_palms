from pathlib import Path

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
DATASET_DIR = Path("../dataset/DBII")
OUTPUT_CSV = Path("labels.csv")

IMG_SCALES = [(128,128),(64,64),(32,32)]
BLOCK_SIZE = 32
GAUSSIAN_BLUR = 5
USE_HOG = True
USE_GABOR = True
USE_LBP = True
MAX_K_CLUSTERS = 5
CONFIDENCE_THRESHOLD = 5
MAX_SAMPLES_PER_CLASS = 200