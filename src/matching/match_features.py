import os
import json
import logging
from itertools import product
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import yaml
from colorama import Fore, Style

# ====================================================
# LOGGING SETUP
# ====================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="data/metadata/matching.log",
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ====================================================
# UTILITY
# ====================================================
def console_step(title: str):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

# ====================================================
# MATCHING MINUTIAE
# ====================================================
def match_minutiae_array(coords_a: np.ndarray, coords_b: np.ndarray, sigma: float, use_type: bool = True) -> float:
    """Calcola il matching score tra due set di minutiae già caricati in memoria."""
    if coords_a.size == 0 or coords_b.size == 0:
        return 0.0

    # Centrare i set
    centroid_a = coords_a[:, :2].mean(axis=0)
    centroid_b = coords_b[:, :2].mean(axis=0)
    coords_a[:, :2] -= centroid_a
    coords_b[:, :2] -= centroid_b

    score = 0.0
    for x_a, y_a, t_a in coords_a:
        dists = np.sqrt((coords_b[:, 0] - x_a) ** 2 + (coords_b[:, 1] - y_a) ** 2)
        type_match = coords_b[:, 2] == t_a if use_type else np.ones(len(coords_b), dtype=bool)
        match_scores = np.exp(-(dists ** 2) / (2 * sigma ** 2)) * type_match
        score += match_scores.max()
    score /= len(coords_a)
    return min(score, 1.0)

def load_dataset(minutiae_base: str) -> Dict[str, List[np.ndarray]]:
    """Carica tutte le minutiae in memoria e converte in array numpy."""
    dataset = {}
    for root, dirs, files in os.walk(minutiae_base):
        for f in files:
            if f.endswith("_minutiae.json"):
                user_id = f.split("_")[0]
                path = os.path.join(root, f)
                try:
                    with open(path, "r") as fin:
                        minutiae = json.load(fin)
                    coords = np.array([[m["x"], m["y"], 0 if m["type"] == "ending" else 1] for m in minutiae], dtype=np.float64)
                    dataset.setdefault(user_id, []).append(coords)
                except Exception as e:
                    logging.warning(f"Errore caricando {path}: {e}")
    return dataset

# ====================================================
# WORKERS
# ====================================================
def frr_worker(args):
    user_id, samples, sigma, use_type, match_threshold = args
    false_rejects = 0
    total = 0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            score = match_minutiae_array(samples[i], samples[j], sigma=sigma, use_type=use_type)
            if score < match_threshold:
                false_rejects += 1
            total += 1
    return false_rejects, total

def far_worker(args):
    samples_i, samples_j, sigma, use_type, match_threshold = args
    false_accepts = 0
    total = 0
    for s1, s2 in product(samples_i, samples_j):
        score = match_minutiae_array(s1, s2, sigma=sigma, use_type=use_type)
        if score >= match_threshold:
            false_accepts += 1
        total += 1
    return false_accepts, total

# ====================================================
# FRR/FAR
# ====================================================
def compute_frr(dataset, sigma, use_type, match_threshold, max_workers=None):
    console_step("Calcolo FRR")
    tasks = [(user_id, samples, sigma, use_type, match_threshold) for user_id, samples in dataset.items()]
    total_false_rejects = 0
    total_comparisons = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fr, tot in tqdm(executor.map(frr_worker, tasks), total=len(tasks), desc="FRR Matching", ncols=90):
            total_false_rejects += fr
            total_comparisons += tot

    frr = total_false_rejects / total_comparisons if total_comparisons else 0.0
    print(f"{Fore.GREEN}✔ FRR calcolato: {frr:.4f}{Style.RESET_ALL}")
    logging.info(f"FRR: {frr:.4f}")
    return frr

def compute_far(dataset, sigma, use_type, match_threshold, max_workers=None):
    console_step("Calcolo FAR")
    users = list(dataset.keys())
    tasks = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            tasks.append((dataset[users[i]], dataset[users[j]], sigma, use_type, match_threshold))

    total_false_accepts = 0
    total_comparisons = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fa, tot in tqdm(executor.map(far_worker, tasks), total=len(tasks), desc="FAR Matching", ncols=90):
            total_false_accepts += fa
            total_comparisons += tot

    far = total_false_accepts / total_comparisons if total_comparisons else 0.0
    print(f"{Fore.GREEN}✔ FAR calcolato: {far:.4f}{Style.RESET_ALL}")
    logging.info(f"FAR: {far:.4f}")
    return far

# ====================================================
# MAIN
# ====================================================
def main(config_path="config/config_matching.yml"):
    console_step("Caricamento Config")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    minutiae_base = cfg.get("minutiae_base", "dataset/processed/minutiae")
    dist_threshold = cfg.get("dist_threshold", 25.0)
    sigma = cfg.get("sigma", dist_threshold / 2.0)
    use_type = cfg.get("use_type", True)
    match_threshold = cfg.get("match_threshold", 0.5)
    max_workers = cfg.get("max_workers", 8)

    print("Caricamento dataset...")
    dataset = load_dataset(minutiae_base)
    print(f"Utenti caricati: {len(dataset)}")
    logging.info(f"Dataset caricato con {len(dataset)} utenti")

    frr = compute_frr(dataset, sigma, use_type, match_threshold, max_workers)
    far = compute_far(dataset, sigma, use_type, match_threshold, max_workers)

    console_step("Matching completato")
    print(f"{Fore.CYAN}✨ FRR = {frr:.4f}, FAR = {far:.4f} ✨{Style.RESET_ALL}")
    logging.info(f"Matching completato: FRR={frr:.4f}, FAR={far:.4f}")

# ====================================================
# ENTRYPOINT
# ====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fingerprint/Minutiae Matching")
    parser.add_argument("--config", type=str, default="config/config_matching.yml")
    args = parser.parse_args()

    console_step("Avvio Matching Minutiae")
    main(config_path=args.config)
