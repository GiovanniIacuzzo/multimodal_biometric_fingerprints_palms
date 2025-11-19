from tqdm import tqdm
from src.matching.match import match_minutiae_pair
from src.matching.utils import console_step
from concurrent.futures import ProcessPoolExecutor
from colorama import Fore, Style
import logging

def frr_worker(args):
    user_id, samples, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers = args

    genuine_scores = []

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            res = match_minutiae_pair(
                samples[i], samples[j],
                dist_thresh=dist_thresh,
                orient_thresh_deg=orient_thresh_deg,
                use_type=use_type,
                ransac_iter=ransac_iter,
                min_inliers=min_inliers
            )

            score = res["final_score"]
            genuine_scores.append(score)

    return genuine_scores


def compute_frr(dataset, dist_thresh, orient_thresh_deg, use_type,
                ransac_iter, min_inliers, match_threshold, max_workers=None):

    console_step("Calcolo FRR")
    tasks = [
        (user_id, samples, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers)
        for user_id, samples in dataset.items()
    ]

    all_genuine_scores = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for scores in tqdm(executor.map(frr_worker, tasks), total=len(tasks),
                           desc="FRR Matching", ncols=90):
            all_genuine_scores.extend(scores)

    total_comparisons = len(all_genuine_scores)
    false_rejects = sum(s < match_threshold for s in all_genuine_scores)
    frr = false_rejects / total_comparisons if total_comparisons else 0.0

    print(f"{Fore.GREEN}✔ FRR calcolato: {frr:.4f}{Style.RESET_ALL}")
    logging.info(f"FRR: {frr:.4f}")

    return frr, all_genuine_scores
