from tqdm import tqdm
from src.matching.match import match_minutiae_pair
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import logging
import csv
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
def compute_frr(dataset,
                dist_thresh,
                orient_thresh_deg,
                use_type,
                ransac_iter,
                min_inliers,
                stop_inlier_ratio=0.15,
                max_workers=1,
                demo=False):

    tasks = []
    for user_id, samples in dataset.items():
        if len(samples) < 2:
            continue

        # tutte le coppie genuine
        pairs = list(combinations(samples, 2))

        # DEMO: limita drasticamente il numero di match
        if demo:
            pairs = pairs[:3]  # max 3 confronti per utente

        for a, b in pairs:
            tasks.append((a, b))

    genuine_scores = []
    match_log = open("logs/genuine_match_stats.csv", "w", newline="")
    writer = csv.writer(match_log)
    writer.writerow([
        "user_id", "idx1", "idx2",
        "score",
        "num_inliers",
        "num_outliers",
        "rotation_deg",
        "translation_x",
        "translation_y"
    ])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for a, b in tasks:
            futures.append(ex.submit(
                match_minutiae_pair,
                a, b,
                dist_thresh=dist_thresh,
                orient_thresh_deg=orient_thresh_deg,
                use_type=use_type,
                ransac_iter=ransac_iter if not demo else 50,
                min_inliers=min_inliers if not demo else 3,
                stop_inlier_ratio=stop_inlier_ratio,
                cross_check=True
            ))

        for f in tqdm(futures, desc="FRR", ncols=90):
            res = f.result()
            score = float(res["final_score"])
            genuine_scores.append(score)

            writer.writerow([
                res.get("user_id", "N/A"),
                res.get("sample_a", -1),
                res.get("sample_b", -1),
                score,
                res.get("inliers", 0),
                res.get("outliers", 0),
                res.get("rotation_deg", 0.0),
                res.get("tx", 0.0),
                res.get("ty", 0.0)
            ])

    match_log.close()
    return genuine_scores
