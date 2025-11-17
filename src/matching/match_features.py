import os
import json
import logging
from itertools import product
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import yaml

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================
# MATCHING MINUTIAE TOLLERANTE
# ============================================================
def match_minutiae(file_a: str, file_b: str, dist_threshold: float, use_type: bool = True, sigma: float = None) -> float:
    try:
        with open(file_a, "r") as f:
            minutiae_a: List[Dict] = json.load(f)
        with open(file_b, "r") as f:
            minutiae_b: List[Dict] = json.load(f)
    except Exception as e:
        logging.warning(f"Errore caricando file minutiae: {e}")
        return 0.0

    if not minutiae_a or not minutiae_b:
        return 0.0

    # Qui forziamo float64
    coords_a = np.array([[m["x"], m["y"], 0 if m["type"]=="ending" else 1] for m in minutiae_a], dtype=np.float64)
    coords_b = np.array([[m["x"], m["y"], 0 if m["type"]=="ending" else 1] for m in minutiae_b], dtype=np.float64)

    # Allineamento centrate
    centroid_a = coords_a[:, :2].mean(axis=0)
    centroid_b = coords_b[:, :2].mean(axis=0)
    coords_a[:, :2] -= centroid_a
    coords_b[:, :2] -= centroid_b

    if sigma is None:
        sigma = dist_threshold / 2.0

    score = 0.0
    for x_a, y_a, t_a in coords_a:
        dists = np.sqrt((coords_b[:,0]-x_a)**2 + (coords_b[:,1]-y_a)**2)
        type_match = coords_b[:,2] == t_a if use_type else np.ones(len(coords_b), dtype=bool)
        match_scores = np.exp(-(dists**2)/(2*sigma**2)) * type_match
        score += match_scores.max()

    score /= len(coords_a)
    return min(score, 1.0)


# ============================================================
# UTILITY PER CARICAMENTO MINUTIAE
# ============================================================
def load_dataset(minutiae_base: str) -> Dict[str, List[str]]:
    dataset = {}
    for root, dirs, files in os.walk(minutiae_base):
        for f in files:
            if f.endswith("_minutiae.json"):
                user_id = f.split("_")[0]
                path = os.path.join(root, f)
                dataset.setdefault(user_id, []).append(path)
    return dataset

# ============================================================
# WORKER GLOBALI PER PARALLELIZZAZIONE
# ============================================================
def frr_worker(args):
    user_id, samples, dist_threshold, use_type, sigma = args
    false_rejects = 0
    total = 0
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            score = match_minutiae(samples[i], samples[j], dist_threshold=dist_threshold, use_type=use_type, sigma=sigma)
            if score < 0.5:  # soglia di accettazione
                false_rejects += 1
            total += 1
    return false_rejects, total

def far_worker(args):
    samples_i, samples_j, dist_threshold, use_type, sigma = args
    false_accepts = 0
    total = 0
    for s1, s2 in product(samples_i, samples_j):
        score = match_minutiae(s1, s2, dist_threshold=dist_threshold, use_type=use_type, sigma=sigma)
        if score >= 0.5:
            false_accepts += 1
        total += 1
    return false_accepts, total

# ============================================================
# CALCOLO FRR
# ============================================================
def compute_frr(dataset: Dict[str, List[str]], dist_threshold: float, use_type: bool, sigma: float, max_workers: int = None) -> float:
    tasks = [(user_id, samples, dist_threshold, use_type, sigma) for user_id, samples in dataset.items()]
    total_false_rejects = 0
    total_comparisons = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fr, tot in tqdm(executor.map(frr_worker, tasks), total=len(tasks), desc="FRR Matching"):
            total_false_rejects += fr
            total_comparisons += tot

    frr = total_false_rejects / total_comparisons if total_comparisons else 0.0
    return frr

# ============================================================
# CALCOLO FAR
# ============================================================
def compute_far(dataset: Dict[str, List[str]], dist_threshold: float, use_type: bool, sigma: float, max_workers: int = None) -> float:
    users = list(dataset.keys())
    tasks = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            tasks.append((dataset[users[i]], dataset[users[j]], dist_threshold, use_type, sigma))

    total_false_accepts = 0
    total_comparisons = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fa, tot in tqdm(executor.map(far_worker, tasks), total=len(tasks), desc="FAR Matching"):
            total_false_accepts += fa
            total_comparisons += tot

    far = total_false_accepts / total_comparisons if total_comparisons else 0.0
    return far

# ============================================================
# MAIN
# ============================================================
def main(config_path: str = "config/config_matching.yml"):
    # Carica configurazione
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    minutiae_base = cfg.get("minutiae_base", "dataset/processed/minutiae")
    dist_threshold = cfg.get("dist_threshold", 15.0)
    use_type = cfg.get("use_type", True)
    sigma = cfg.get("sigma", dist_threshold/2.0)
    max_workers = cfg.get("max_workers", 8)

    print("Caricamento dataset...")
    dataset = load_dataset(minutiae_base)
    print(f"Utenti caricati: {len(dataset)}")

    print("Calcolo FRR...")
    frr = compute_frr(dataset, dist_threshold=dist_threshold, use_type=use_type, sigma=sigma, max_workers=max_workers)
    print(f"FRR = {frr:.4f}")

    print("Calcolo FAR...")
    far = compute_far(dataset, dist_threshold=dist_threshold, use_type=use_type, sigma=sigma, max_workers=max_workers)
    print(f"FAR = {far:.4f}")

# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fingerprint/Minutiae Matching")
    parser.add_argument("--config", type=str, default="config/config_matching.yml")
    args = parser.parse_args()

    main(config_path=args.config)
