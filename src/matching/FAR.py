from tqdm import tqdm
from src.matching.match import match_minutiae_pair
from src.matching.utils import console_step
from concurrent.futures import ProcessPoolExecutor
from colorama import Fore, Style
import logging

def far_worker_batch(args):
    samples_i, samples_j, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers = args
    batch_scores = []
    for s1 in samples_i:
        for s2 in samples_j:
            res = match_minutiae_pair(
                s1, s2,
                dist_thresh=dist_thresh,
                orient_thresh_deg=orient_thresh_deg,
                use_type=use_type,
                ransac_iter=ransac_iter,
                min_inliers=min_inliers
            )
            batch_scores.append(res["final_score"])
    return batch_scores


def compute_far(dataset, dist_thresh, orient_thresh_deg, use_type,
                ransac_iter, min_inliers, match_threshold, max_workers=None):

    console_step("Calcolo FAR")
    users = list(dataset.keys())
    tasks = []

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            tasks.append((
                dataset[users[i]],
                dataset[users[j]],
                dist_thresh,
                orient_thresh_deg,
                use_type,
                ransac_iter,
                min_inliers
            ))

    all_impostor_scores = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_scores in tqdm(
            executor.map(far_worker_batch, tasks, chunksize=1),
            total=len(tasks),
            desc="FAR Matching",
            ncols=90
        ):
            all_impostor_scores.extend(batch_scores)

    total_comparisons = len(all_impostor_scores)
    false_accepts = sum(s >= match_threshold for s in all_impostor_scores)
    far = false_accepts / total_comparisons if total_comparisons else 0.0

    print(f"{Fore.GREEN}✔ FAR calcolato: {far:.4f}{Style.RESET_ALL}")
    logging.info(f"FAR: {far:.4f}")

    return far, all_impostor_scores
