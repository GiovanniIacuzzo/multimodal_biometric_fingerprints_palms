import numpy as np
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
from colorama import Fore, Style
import matplotlib.pyplot as plt
from src.matching.match import match_minutiae_pair
from src.matching.utils import console_step

def score_worker(args):
    kind, mins_a, mins_b, dist_thresh, orient_thresh_deg, use_type, ransac_iter, min_inliers = args

    res = match_minutiae_pair(
        mins_a, mins_b,
        dist_thresh=dist_thresh,
        orient_thresh_deg=orient_thresh_deg,
        use_type=use_type,
        ransac_iter=ransac_iter,
        min_inliers=min_inliers
    )

    score = res["final_score"]
    return kind, score

def compute_all_scores(dataset,
                       dist_threshold,
                       orient_thresh_deg,
                       use_type,
                       ransac_iter,
                       min_inliers,
                       max_workers=None):

    console_step("Calcolo punteggi Genuine & Impostor")

    tasks = []

    users = list(dataset.keys())

    # Genuine pairs
    for user in users:
        samples = dataset[user]
        for a, b in combinations(samples, 2):
            tasks.append(("genuine", a, b, dist_threshold, orient_thresh_deg,
                          use_type, ransac_iter, min_inliers))

    # Impostor pairs
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            s_i = dataset[users[i]]
            s_j = dataset[users[j]]
            for a, b in product(s_i, s_j):
                tasks.append(("impostor", a, b, dist_threshold, orient_thresh_deg,
                              use_type, ransac_iter, min_inliers))

    genuine_scores = []
    impostor_scores = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for kind, score in tqdm(
            executor.map(score_worker, tasks),
            total=len(tasks),
            desc="Calcolo punteggi",
            ncols=90
        ):
            if kind == "genuine":
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    genuine_scores = np.array(genuine_scores, dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    print(f"{Fore.GREEN}✔ Raccolti punteggi: Genuine={len(genuine_scores)}, Impostor={len(impostor_scores)}{Style.RESET_ALL}")
    logging.info(f"Genuine scores: {len(genuine_scores)}, Impostor scores: {len(impostor_scores)}")

    return genuine_scores, impostor_scores

def compute_roc_from_scores(genuine_scores,
                            impostor_scores,
                            thresholds=None):

    if thresholds is None:
        thresholds = np.linspace(0, 1, 51)

    frrs = []
    fars = []

    for th in thresholds:
        # FRR: genuine rifiutati
        frr = np.mean(genuine_scores < th)

        # FAR: impostor accettati
        far = np.mean(impostor_scores >= th)

        frrs.append(frr)
        fars.append(far)

    return thresholds, np.array(frrs), np.array(fars)

def plot_roc(genuine_scores, impostor_scores, save_path="ROC.png"):
    thresholds = np.linspace(0, 1, 101)
    fpr_list = []
    tpr_list = []

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    for thr in thresholds:
        tpr = np.sum(genuine_scores >= thr) / len(genuine_scores)
        fpr = np.sum(impostor_scores >= thr) / len(impostor_scores)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = -np.trapz(tpr_list, fpr_list)

    plt.figure(figsize=(6,6))
    plt.plot(fpr_list, tpr_list, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1-FRR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    return fpr_list, tpr_list, auc


def compute_roc(dataset,
                dist_threshold,
                orient_thresh_deg,
                use_type,
                ransac_iter,
                min_inliers,
                max_workers=None,
                thresholds=None):

    genuine_scores, impostor_scores = compute_all_scores(
        dataset, dist_threshold, orient_thresh_deg,
        use_type, ransac_iter, min_inliers, max_workers
    )

    return compute_roc_from_scores(genuine_scores, impostor_scores, thresholds)
