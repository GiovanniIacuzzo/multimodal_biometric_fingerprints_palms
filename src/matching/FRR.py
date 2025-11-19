from tqdm import tqdm
from src.matching.match import match_minutiae_pair
from src.matching.utils import console_step
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, Style
import logging
import numpy as np
import os

# ----------------------------
# Configurazione Logger
# ----------------------------
LOGFILE = "data/metadata/matching.log"
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

logging.basicConfig(
    filename=LOGFILE,
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

DEBUG = True

# ----------------------------
# Worker FRR
# ----------------------------
def frr_worker(args):
    user_id, samples, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers = args
    genuine_scores = []

    logger.info(f"[USER {user_id}] Numero campioni: {len(samples)}")

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):

            logger.debug(f"[USER {user_id}] match {i} vs {j}")
            res = match_minutiae_pair(
                samples[i], samples[j],
                dist_thresh=dist_thresh,
                orient_thresh_deg=orient_thresh_deg,
                use_type=use_type,
                ransac_iter=ransac_iter,
                min_inliers=min_inliers
            )

            score = res.get("final_score", 0.0)
            genuine_scores.append(score)

            logger.debug(f"[USER {user_id}] score: {score:.4f}")

    if genuine_scores:
        logger.info(
            f"[USER {user_id}] min={np.min(genuine_scores):.4f}, "
            f"max={np.max(genuine_scores):.4f}, "
            f"mean={np.mean(genuine_scores):.4f}, "
            f"std={np.std(genuine_scores):.4f}, "
            f"num={len(genuine_scores)}"
        )

    return genuine_scores

# ----------------------------
# Compute FRR
# ----------------------------
def compute_frr(dataset, dist_thresh, orient_thresh_deg, use_type,
                ransac_iter, min_inliers, match_threshold, max_workers=None):

    console_step("Calcolo FRR")
    tasks = [
        (user_id, samples, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers)
        for user_id, samples in dataset.items()
    ]

    all_genuine_scores = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for scores in tqdm(executor.map(frr_worker, tasks), total=len(tasks),
                           desc="FRR Matching", ncols=90):
            all_genuine_scores.extend(scores)

    # ============================
    # LOG GLOBALI DIAGNOSTICI
    # ============================
    if all_genuine_scores:
        zeros = sum(s == 0 for s in all_genuine_scores)
        ones = sum(s == 1 for s in all_genuine_scores)
        nans = sum(np.isnan(s) for s in all_genuine_scores)

        logger.info(f"[GLOBAL] Numero totale confronti: {len(all_genuine_scores)}")
        logger.info(f"[GLOBAL] Zero scores: {zeros}, One scores: {ones}, NaN scores: {nans}")
        logger.info(f"[GLOBAL] min={np.min(all_genuine_scores):.4f}, "
                    f"max={np.max(all_genuine_scores):.4f}, "
                    f"mean={np.mean(all_genuine_scores):.4f}, "
                    f"std={np.std(all_genuine_scores):.4f}")
        logger.info("[GLOBAL] 20 punteggi più bassi: " + str(sorted(all_genuine_scores)[:20]))
        logger.info("[GLOBAL] 20 punteggi più alti: " + str(sorted(all_genuine_scores)[-20:]))

        # backup print se logging non funziona
        if DEBUG:
            print(f"[DEBUG] {len(all_genuine_scores)} confronti, min={np.min(all_genuine_scores):.4f}, max={np.max(all_genuine_scores):.4f}")

    # FRR vero
    total_comparisons = len(all_genuine_scores)
    false_rejects = sum(s < match_threshold for s in all_genuine_scores)
    frr = false_rejects / total_comparisons if total_comparisons else 0.0

    print(f"{Fore.GREEN}✔ FRR calcolato: {frr:.4f}{Style.RESET_ALL}")
    logger.info(f"FRR: {frr:.4f}, threshold={match_threshold}")

    return frr, all_genuine_scores
