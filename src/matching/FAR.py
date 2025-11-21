from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import random
from src.matching.match import match_minutiae_pair

def far_worker_batch(args):
    samples_a, samples_b, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers, stop_inlier_ratio = args

    batch_scores = []
    for a in samples_a:
        for b in samples_b:
            res = match_minutiae_pair(
                a, b,
                dist_thresh=dist_thresh,
                orient_thresh_deg=orient_thresh_deg,
                use_type=use_type,
                ransac_iter=ransac_iter,
                min_inliers=min_inliers,
                stop_inlier_ratio=stop_inlier_ratio,
                cross_check=True,
                thread_workers=1  # dentro ogni processo non usiamo thread
            )
            batch_scores.append(float(res["final_score"]))
    return batch_scores

def sample_impostor_pairs(users, sample_size=100):
    """
    Campiona le coppie impostori per velocizzare il calcolo FAR
    """
    pairs = []
    for u1 in users:
        # prendi sample_size utenti diversi da u1
        u2_sample = random.sample([u for u in users if u != u1], min(sample_size, len(users)-1))
        for u2 in u2_sample:
            pairs.append((u1, u2))
    return pairs

def compute_far(dataset,
                dist_thresh,
                orient_thresh_deg,
                use_type,
                ransac_iter,
                min_inliers,
                stop_inlier_ratio=0.15,
                max_workers=4,
                impostor_sample_size=100):
    
    users = list(dataset.keys())
    task_pairs = sample_impostor_pairs(users, sample_size=impostor_sample_size)

    tasks = []
    for u1, u2 in task_pairs:
        tasks.append((
            dataset[u1],
            dataset[u2],
            dist_thresh,
            orient_thresh_deg,
            use_type,
            ransac_iter,
            min_inliers,
            stop_inlier_ratio
        ))

    impostor_scores = []

    print(f"Submitting {len(tasks)} FAR tasks...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(far_worker_batch, args): args for args in tasks}

        for f in tqdm(as_completed(futures), total=len(futures), desc="FAR", ncols=90):
            impostor_scores.extend(f.result())

    return impostor_scores
