import math
import logging
from itertools import combinations
from typing import List, Dict, Tuple
from src.db.database import load_minutiae_from_db, save_matching_result

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
def euclidean_distance(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def angle_between(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    return math.atan2(p2[1]-p1[1], p2[0]-p1[0])

# ========================
# Creazione coppie
# ========================
def build_pairs(minutiae: List[Minutia]) -> List[Tuple[Tuple[int,int], float, float]]:
    pairs = []
    for (i1, m1), (i2, m2) in combinations(enumerate(minutiae), 2):
        if m1['type'] != m2['type']:
            continue
        d = euclidean_distance((m1['x'], m1['y']), (m2['x'], m2['y']))
        a = angle_between((m1['x'], m1['y']), (m2['x'], m2['y']))
        pairs.append(((i1, i2), d, a))
    logger.debug(f"Generate {len(pairs)} valid pairs.")
    return pairs

# ========================
# Matching coppie
# ========================
def match_pairs(pairs1, pairs2, dist_thresh=25.0, angle_thresh=0.5) -> int:
    matched = 0
    for (_, d1, a1) in pairs1:
        for (_, d2, a2) in pairs2:
            if abs(d1 - d2) <= dist_thresh and abs(a1 - a2) <= angle_thresh:
                matched += 1
                break
    logger.debug(f"Matched {matched} pairs (thresholds: dist={dist_thresh}, angle={angle_thresh})")
    return matched

# ========================
# Matching due immagini
# ========================
def match_two_images(image_id_a: int, image_id_b: int) -> float:
    logger.info(f"Inizio matching: immagine {image_id_a} vs {image_id_b}")
    minutiae1 = load_minutiae_from_db(image_id_a)
    minutiae2 = load_minutiae_from_db(image_id_b)

    if len(minutiae1) < 2 or len(minutiae2) < 2:
        logger.warning(f"Immagini {image_id_a} o {image_id_b} con minutiae insufficienti "
                       f"({len(minutiae1)}, {len(minutiae2)}).")
        return 0.0

    logger.debug(f"Immagine {image_id_a}: {len(minutiae1)} minutiae | "
                 f"Immagine {image_id_b}: {len(minutiae2)} minutiae")

    pairs1 = build_pairs(minutiae1)
    pairs2 = build_pairs(minutiae2)
    matched_pairs = match_pairs(pairs1, pairs2)
    total_pairs = max(len(pairs1), len(pairs2))
    score = matched_pairs / total_pairs if total_pairs > 0 else 0.0

    logger.info(f"Completato matching {image_id_a} vs {image_id_b} → score={score:.4f} "
                f"(matched {matched_pairs}/{total_pairs})")
    return score

# ========================
# Batch matching
# ========================
def batch_match(image_ids: List[int], method: str = "pair_matching") -> Dict[Tuple[int,int], float]:
    """
    Esegue matching tra tutte le coppie di immagini specificate.
    Salva i risultati in DB.
    """
    results = {}
    total = len(image_ids)
    logger.info(f"Avvio batch matching su {total} immagini ({total*(total-1)//2} confronti previsti)")

    for i, a_id in enumerate(image_ids):
        for b_id in image_ids[i+1:]:
            try:
                score = match_two_images(a_id, b_id)
                results[(a_id, b_id)] = score
                save_matching_result(a_id, b_id, score, method)
            except Exception as e:
                logger.error(f"Errore durante il matching {a_id} vs {b_id}: {e}")
                continue

    logger.info(f"Batch matching completato con successo ({len(results)} confronti salvati)")
    return results
