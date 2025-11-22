import os
import json
import logging
from typing import Dict, List
import numpy as np
import yaml

from src.matching.FRR import compute_frr
from src.matching.FAR import compute_far
from src.matching.ROC import plot_roc
from src.matching.utils import console_step, report_scores, evaluate_frr_across_thresholds, evaluate_far_across_thresholds, compute_minutiae_statistics

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

    for root, dirs, files in os.walk(minutiae_base):
        for f in files:
            if f.endswith("_minutiae.json"):
                user_id = f.split("_")[0]
                full_path = os.path.join(root, f)
                files_per_user.setdefault(user_id, []).append(full_path)

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
                    t = 0 if m.get("type", "ending") == "ending" else 1
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
def main(config_path="config/config_matching.yml", demo=False):
    console_step("Caricamento Configurazione")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    minutiae_base = cfg.get("minutiae_base", "dataset/processed/minutiae")

    if cfg.get("deterministic", True):
        np.random.seed(42)
        logging.info("Deterministic mode: ON (seed=42)")

    # ----------------------------------------------
    # DEMO MODE SETTINGS
    # ----------------------------------------------
    if demo:
        print("\n⚡ DEMO MODE ATTIVO → pipeline ultrarapida ⚡\n")

        demo_settings = {
            "max_per_user": 2,          # invece di 5–10
            "frr_ransac": 500,          # invece di 300
            "far_ransac": 500,          # invece di 300
            "frr_min_inliers": 5,
            "far_min_inliers": 5,
            "num_points": 30            # invece di 50
        }
    else:
        demo_settings = {
            "max_per_user": 2,
            "frr_ransac": 300,
            "far_ransac": 300,
            "frr_min_inliers": 6,
            "far_min_inliers": 12,
            "num_points": 50
        }

    # ----------------------------------------------
    # Caricamento dataset
    # ----------------------------------------------
    console_step("Caricamento Dataset")
    dataset = load_dataset(minutiae_base, max_per_user=demo_settings["max_per_user"])
    print(f"Utenti caricati: {len(dataset)}")
    compute_minutiae_statistics(dataset, output_file="logs/minutiae_stats.csv")


    # ----------------------------------------------
    # Calcolo FRR
    # ----------------------------------------------
    console_step("Calcolo FRR")
    genuine_scores = compute_frr(
        dataset,
        dist_thresh=30,
        orient_thresh_deg=30,
        use_type=True,
        ransac_iter=demo_settings["frr_ransac"],
        min_inliers=demo_settings["frr_min_inliers"],
    )

    report_scores("REPORT FRR (Genuine Scores)", genuine_scores)
    th_frr, frr = evaluate_frr_across_thresholds(genuine_scores, num_points=demo_settings["num_points"])

    # ----------------------------------------------
    # Calcolo FAR
    # ----------------------------------------------
    console_step("Calcolo FAR")
    impostor_scores = compute_far(
        dataset,
        dist_thresh=15,
        orient_thresh_deg=10,
        use_type=True,
        ransac_iter=demo_settings["far_ransac"],
        min_inliers=demo_settings["far_min_inliers"],
        demo=demo
    )

    report_scores("REPORT FAR (Impostor Scores)", impostor_scores)
    th_far, far = evaluate_far_across_thresholds(impostor_scores, num_points=demo_settings["num_points"])

    # ----------------------------------------------
    # ROC Curve
    # ----------------------------------------------
    console_step("Generazione ROC")
    plot_roc(th_far, far_values=far, frr_values=frr, title="ROC (FAR vs FRR)")

    print("\n✨ Matching completato ✨\n")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fingerprint/Minutiae Matching")
    parser.add_argument("--config", type=str, default="config/config_matching.yml")
    parser.add_argument("--demo", action="store_true", help="Esegui in modalità DEMO ultrarapida")
    args = parser.parse_args()

    console_step("Avvio Matching Minutiae")
    main(config_path=args.config, demo=args.demo)
