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

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="data/metadata/matching.log",
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_dataset(minutiae_base: str, max_per_user: int = None) -> Dict[str, List[np.ndarray]]:
    dataset = {}
    files_per_user = {}

    # Raggruppa i file JSON per utente
    for root, dirs, files in os.walk(minutiae_base):
        for f in files:
            if f.endswith("_minutiae.json"):
                user_id = f.split("_")[0]
                path = os.path.join(root, f)
                files_per_user.setdefault(user_id, []).append(path)

    # Ordina i file e carica solo i primi max_per_user
    for user_id, paths in files_per_user.items():
        paths_sorted = sorted(paths)  # ordina alfabeticamente
        if max_per_user is not None:
            paths_sorted = paths_sorted[:max_per_user]

        # print(f"\nUtente {user_id} - caricati {len(paths_sorted)} file JSON:")
        # for p in paths_sorted:
        #     print(f"  {p}")

        dataset[user_id] = []
        for path in paths_sorted:
            try:
                with open(path, "r") as fin:
                    minutiae = json.load(fin)
                arr = []
                for m in minutiae:
                    t = 0 if m.get("type","ending") == "ending" else 1
                    arr.append([
                        float(m["x"]),
                        float(m["y"]),
                        float(t),
                        float(m.get("orientation", 0.0)),
                        float(m.get("quality", 0.0)),
                        float(m.get("coherence", 0.0)),
                        float(m.get("angular_stability", 0.0))
                    ])
                coords = np.array(arr, dtype=np.float64)
                dataset[user_id].append(coords)
            except Exception as e:
                logging.warning(f"Errore caricando {path}: {e}")

    return dataset


# ------------------------------------------------------------
# MAIN CLI
# ------------------------------------------------------------
def main(config_path="config/config_matching.yml"):
    console_step("Caricamento Config")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    minutiae_base = cfg.get("minutiae_base", "dataset/processed/minutiae")
    dist_threshold = cfg.get("dist_threshold", 25.0)
    orient_thresh_deg = cfg.get("orient_thresh_deg", 20.0)
    use_type = cfg.get("use_type", True)
    ransac_iter = cfg.get("ransac_iter", 300)
    min_inliers = cfg.get("min_inliers", 6)
    match_threshold = cfg.get("match_threshold", 0.4)
    max_workers = cfg.get("max_workers", 8)

    print("Caricamento dataset...")
    dataset = load_dataset(minutiae_base, max_per_user=5)
    print(f"Utenti caricati: {len(dataset)}")
    logging.info(f"Dataset caricato con {len(dataset)} utenti")

    frr, genuine_scores = compute_frr(
        dataset, dist_threshold, orient_thresh_deg,
        use_type, ransac_iter, min_inliers, match_threshold, max_workers
    )

    far, impostor_scores = compute_far(
        dataset, dist_threshold, orient_thresh_deg,
        use_type, ransac_iter, min_inliers, match_threshold, max_workers
    )
    
    console_step("Matching completato")
    print(f"{Fore.CYAN}✨ FRR = {frr:.4f}, FAR = {far:.4f} ✨{Style.RESET_ALL}")
    logging.info(f"Matching completato: FRR={frr:.4f}, FAR={far:.4f}")

    console_step("Generazione ROC Curve")

    fpr, tpr, roc_auc = plot_roc(
        genuine_scores,
        impostor_scores,
        save_path="data/metadata/ROC_curve.png"
    )

    print(f"{Fore.GREEN}ROC AUC = {roc_auc:.4f}{Style.RESET_ALL}")
    logging.info(f"ROC curve generata AUC={roc_auc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fingerprint/Minutiae Matching")
    parser.add_argument("--config", type=str, default="config/config_matching.yml")
    args = parser.parse_args()

    console_step("Avvio Matching Minutiae")
    main(config_path=args.config)
