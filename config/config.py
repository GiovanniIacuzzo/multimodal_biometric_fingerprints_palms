import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "DBII")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

DEBUG_DIR = os.path.join(FEATURES_DIR, "debug")

PRUNE_ITERS = 2
PRUNE_AREA = 2
ORIENT_SIGMA = 3.0

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 7
GAUSSIAN_SIGMA = 0.7

SAUVOLA_WIN = 25
SAUVOLA_K = 0.2
LOCAL_PATCH = 64
MIN_OBJ_SIZE = 30
MAX_HOLE_SIZE = 100
MIN_SEGMENT_AREA = 5000

BLOCK_SIZE = 16
ORIENT_SIGMA = 7.0
ENERGY_THRESHOLD = 1e-2
REL_THRESHOLD = 0.2
VIS_SCALE = 8