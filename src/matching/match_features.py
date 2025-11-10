import math
import time
import logging
from itertools import combinations
from typing import List, Dict, Tuple
from tqdm import tqdm
from src.db.database import (
    load_minutiae_from_db,
    save_matching_result,
    get_image_id_by_filename,
)

# ========================
# Configurazione logging
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ========================
# Tipi
# ========================
Minutia = Dict[str, float]

# ========================
# Funzioni di utilità
# ========================
def euclidean_distance(m1: Minutia, m2: Minutia) -> float:
    return math.hypot(m1["x"] - m2["x"], m1["y"] - m2["y"])

def orientation_difference(m1: Minutia, m2: Minutia) -> float:
    """Differenza angolare tra due minutiae"""
    return abs(m1["orientation"] - m2["orientation"])

# ========================
# Creazione coppie
# ========================
def build_pairs(minutiae: List[Minutia]) -> List[Tuple[Minutia, Minutia, float, float]]:
    """
    Genera tutte le coppie compatibili di minutiae con distanza e differenza di orientamento.
    """
    pairs = []
    for m1, m2 in combinations(minutiae, 2):
        if m1["type"] != m2["type"]:
            continue
        dist = euclidean_distance(m1, m2)
        angle_diff = orientation_difference(m1, m2)
        pairs.append((m1, m2, dist, angle_diff))
    return pairs

# ========================
# Matching coppie
# ========================
def match_pairs(
    pairs1: List[Tuple[Minutia, Minutia, float, float]],
    pairs2: List[Tuple[Minutia, Minutia, float, float]],
    dist_thresh: float = 25.0,
    angle_thresh: float = 0.5
) -> int:
    matched = 0
    for _, _, d1, a1 in pairs1:
        for _, _, d2, a2 in pairs2:
            if abs(d1 - d2) <= dist_thresh and abs(a1 - a2) <= angle_thresh:
                matched += 1
                break
    return matched

# ========================
# Matching due immagini
# ========================
def match_two_images(
    image_id_a: int,
    image_id_b: int,
    dist_thresh: float = 25.0,
    angle_thresh: float = 0.5
) -> float:
    """Confronta due immagini tramite i loro ID nel DB."""
    start_time = time.perf_counter()

    minutiae1 = load_minutiae_from_db(image_id_a)
    minutiae2 = load_minutiae_from_db(image_id_b)

    if len(minutiae1) < 2 or len(minutiae2) < 2:
        return 0.0

    pairs1 = build_pairs(minutiae1)
    pairs2 = build_pairs(minutiae2)

    if not pairs1 or not pairs2:
        return 0.0

    matched_pairs = match_pairs(pairs1, pairs2, dist_thresh, angle_thresh)
    total_pairs = max(len(pairs1), len(pairs2))
    score = matched_pairs / total_pairs if total_pairs > 0 else 0.0

    elapsed = time.perf_counter() - start_time
    # logger.debug(f"{image_id_a} vs {image_id_b} | score={score:.4f} | tempo={elapsed*1000:.1f}ms")
    return score

# ========================
# Batch matching
# ========================
def batch_match(
    image_filenames: List[str],
    method: str = "pair_matching",
    dist_thresh: float = 25.0,
    angle_thresh: float = 0.5
) -> Dict[Tuple[str, str], float]:
    """
    Esegue il matching batch tra tutte le immagini elencate (per filename).
    Restituisce un dizionario: {(filename_a, filename_b): score}
    """
    results = {}
    total_pairs = len(image_filenames) * (len(image_filenames) - 1) // 2
    logger.info(f"Avvio batch matching ({total_pairs} confronti)")

    start = time.perf_counter()

    # Creazione lista di tutte le coppie
    pairs = [
        (file_a, file_b)
        for i, file_a in enumerate(image_filenames)
        for file_b in image_filenames[i + 1:]
    ]

    # Barra di avanzamento tqdm
    for file_a, file_b in tqdm(pairs, desc="Matching immagini", unit="coppia"):
        try:
            id_a = get_image_id_by_filename(file_a)
            id_b = get_image_id_by_filename(file_b)

            if id_a is None or id_b is None:
                logger.warning(f"Immagine non trovata nel DB: {file_a} o {file_b}")
                continue

            score = match_two_images(id_a, id_b, dist_thresh, angle_thresh)
            results[(file_a, file_b)] = score

            # Salva risultato nel DB
            save_matching_result(id_a, id_b, score, method)

        except Exception as e:
            logger.error(f"Errore matching {file_a} vs {file_b}: {e}")

    elapsed = time.perf_counter() - start
    if results:
        logger.info(f"Batch completato in {elapsed:.2f}s ({elapsed/len(results):.3f}s per confronto medio).")
    else:
        logger.warning("Nessun confronto completato! Controlla le immagini nel DB.")

    return results