import os
import json
import logging
from typing import Dict, List
import numpy as np
import yaml
from colorama import Fore, Style

from src.matching.FRR import compute_frr
from src.matching.FAR import compute_far
from src.matching.ROC import plot_roc
from src.matching.utils import console_step


# ------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="data/metadata/matching.log",
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ------------------------------------------------------------
# Dataset Loader
# ------------------------------------------------------------
def load_dataset(minutiae_base: str, max_per_user: int = None) -> Dict[str, List[np.ndarray]]:
    dataset = {}
    files_per_user = {}

    # Trova tutti i JSON
    for root, dirs, files in os.walk(minutiae_base):
        for f in files:
            if f.endswith("_minutiae.json"):
                user_id = f.split("_")[0]
                full_path = os.path.join(root, f)
                files_per_user.setdefault(user_id, []).append(full_path)

    # Caricamento minuzie
    for user_id, paths in files_per_user.items():
        paths_sorted = sorted(paths)

        if max_per_user is not None:
            paths_sorted = paths_sorted[:max_per_user]

        user_minutiae = []

        for path in paths_sorted:
            try:
                with open(path, "r") as fin:
                    minutiae = json.load(fin)

                arr = []
                for m in minutiae:
                    # Tipo: ending=0, bifurcation=1
                    t = 0 if m.get("type", "ending") == "ending" else 1

                    # Ordine campi mantenuto fisso
                    arr.append([
                        float(m["x"]),
                        float(m["y"]),
                        float(t),
                        float(m.get("orientation", 0.0)),
                        float(m.get("quality", 0.0)),
                        float(m.get("coherence", 0.0)),
                        float(m.get("angular_stability", 0.0))
                    ])

                user_minutiae.append(np.array(arr, dtype=np.float64))

            except Exception as e:
                logging.warning(f"Errore caricando {path}: {e}")

        dataset[user_id] = user_minutiae

    return dataset


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(config_path="config/config_matching.yml"):
    console_step("Caricamento Configurazione")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Parametri matching ottimizzati
    minutiae_base       = cfg.get("minutiae_base", "dataset/processed/minutiae")
    dist_threshold      = cfg.get("dist_threshold", 22.0)
    orient_thresh_deg   = cfg.get("orient_thresh_deg", 22.0)
    use_type            = cfg.get("use_type", True)
    ransac_iter         = cfg.get("ransac_iter", 800)
    min_inliers         = cfg.get("min_inliers", 7)
    match_threshold     = cfg.get("match_threshold", 0.42)
    max_workers         = cfg.get("max_workers", 8)

    # Determinismo opzionale
    if cfg.get("deterministic", True):
        np.random.seed(42)
        logging.info("Deterministic mode: ON (seed=42)")

    # ----------------------------------------------
    # Caricamento dataset
    # ----------------------------------------------
    console_step("Caricamento Dataset")
    print("Caricamento dataset...")
    dataset = load_dataset(minutiae_base, max_per_user=2)

    print(f"Utenti caricati: {len(dataset)}")
    logging.info(f"Dataset caricato: {len(dataset)} utenti")

    # ----------------------------------------------
    # Calcolo FRR
    # ----------------------------------------------
    console_step("Calcolo FRR")

    frr, genuine_scores = compute_frr(
        dataset,
        dist_threshold,
        orient_thresh_deg,
        use_type,
        ransac_iter,
        min_inliers,
        match_threshold,
        max_workers
    )

    # ----------------------------------------------
    # Calcolo FAR
    # ----------------------------------------------
    console_step("Calcolo FAR")

    far, impostor_scores = compute_far(
        dataset,
        dist_threshold,
        orient_thresh_deg,
        use_type,
        ransac_iter,
        min_inliers,
        match_threshold,
        max_workers
    )

    # ----------------------------------------------
    # Risultati Matching
    # ----------------------------------------------
    console_step("Matching Completato")

    print(f"{Fore.CYAN}✨ FRR = {frr:.4f}, FAR = {far:.4f} ✨{Style.RESET_ALL}")
    logging.info(f"Risultati Matching → FRR={frr:.4f}, FAR={far:.4f}")

    # ----------------------------------------------
    # ROC Curve
    # ----------------------------------------------
    console_step("Generazione ROC Curve")

    fpr_list, tpr_list, auc, eer, eer_threshold = plot_roc(
        genuine_scores,
        impostor_scores,
        save_path="data/metadata/ROC_curve.png"
    )

    print(f"{Fore.GREEN}ROC AUC = {auc:.4f}{Style.RESET_ALL}")
    logging.info(f"ROC Curve generata con AUC={auc:.4f}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fingerprint/Minutiae Matching")
    parser.add_argument("--config", type=str, default="config/config_matching.yml")
    args = parser.parse_args()

    console_step("Avvio Matching Minutiae")
    main(config_path=args.config)
