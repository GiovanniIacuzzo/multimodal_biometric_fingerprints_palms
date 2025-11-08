import json
import csv
import os
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

# ===========================
# Funzioni di caricamento
# ===========================

def load_results(results_path: str) -> Dict[Tuple[str, str], float]:
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"File risultati non trovato: {results_path}")

    results = {}
    if results_path.endswith(".json"):
        with open(results_path, "r") as f:
            raw = json.load(f)
        for key, value in raw.items():
            if isinstance(key, str) and "_vs_" in key:
                u1, u2 = key.split("_vs_")
                results[(u1, u2)] = value
            elif isinstance(key, (list, tuple)) and len(key) == 2:
                results[tuple(key)] = value
            else:
                print(f"[WARN] Chiave JSON non riconosciuta: {key}")
    elif results_path.endswith(".csv"):
        with open(results_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[(row["user1"], row["user2"])] = float(row["score"])
    else:
        raise ValueError("Formato non supportato (solo .json o .csv)")
    return results

# ===========================
# Etichettatura match
# ===========================

def compute_labels(results):
    scores = []
    labels = []
    rows = []

    for key, score in results.items():
        if isinstance(key, tuple):
            u1, u2 = key
        else:
            u1, u2 = key.split("_vs_")

        id1 = u1.split("_")[0]
        id2 = u2.split("_")[0]

        is_genuine = int(id1 == id2)
        labels.append(is_genuine)
        scores.append(score)
        rows.append((u1, u2, score, is_genuine))

    return scores, labels, rows

# ===========================
# Calcolo metriche
# ===========================

def compute_metrics(scores: List[float], labels: List[int], threshold: float = None) -> Dict[str, float]:
    scores = np.array(scores)
    labels = np.array(labels)

    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    FARs, FRRs, thresholds = [], [], []
    total_genuine = np.sum(sorted_labels)
    total_impostor = len(labels) - total_genuine

    for t in sorted_scores:
        FA = np.sum((sorted_scores >= t) & (sorted_labels == 0))
        FR = np.sum((sorted_scores < t) & (sorted_labels == 1))
        FAR = FA / total_impostor if total_impostor > 0 else 0
        FRR = FR / total_genuine if total_genuine > 0 else 0
        FARs.append(FAR)
        FRRs.append(FRR)
        thresholds.append(t)

    idx_eer = np.argmin(np.abs(np.array(FARs) - np.array(FRRs)))
    eer = (FARs[idx_eer] + FRRs[idx_eer]) / 2
    threshold_eer = thresholds[idx_eer]

    if threshold is None:
        threshold = threshold_eer

    preds = (scores >= threshold).astype(int)
    accuracy = np.mean(preds == labels)

    return {
        "EER": round(eer, 4),
        "FAR@EER": round(FARs[idx_eer], 4),
        "FRR@EER": round(FRRs[idx_eer], 4),
        "Accuracy": round(accuracy, 4),
        "Threshold": round(threshold, 4)
    }

# ===========================
# Salvataggio risultati
# ===========================

def save_metrics(metrics: Dict[str, float], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metriche salvate in {output_path}")

def save_detailed_results(rows: List[Tuple[str,str,float,int]], output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user1", "user2", "score", "label"])
        writer.writerows(rows)
    print(f"[INFO] Risultati dettagliati salvati in {output_csv}")

# ===========================
# Plot FAR-FRR Curve
# ===========================

def plot_far_frr_curve(scores: List[float], labels: List[int], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    scores = np.array(scores)
    labels = np.array(labels)

    thresholds = np.sort(scores)
    FARs, FRRs = [], []

    total_genuine = np.sum(labels)
    total_impostor = len(labels) - total_genuine

    for t in thresholds:
        FA = np.sum((scores >= t) & (labels == 0))
        FR = np.sum((scores < t) & (labels == 1))
        FAR = FA / total_impostor if total_impostor > 0 else 0
        FRR = FR / total_genuine if total_genuine > 0 else 0
        FARs.append(FAR)
        FRRs.append(FRR)

    plt.figure(figsize=(6,6))
    plt.plot(FARs, FRRs, lw=2, color='blue', label='FAR vs FRR')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='EER line')
    plt.xlabel("FAR (False Acceptance Rate)")
    plt.ylabel("FRR (False Rejection Rate)")
    plt.title("FAR vs FRR Curve")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "far_frr_curve.png"))
    plt.close()
    print(f"[INFO] FAR-FRR curve salvata in {os.path.join(output_dir, 'far_frr_curve.png')}")

# ===========================
# Funzione principale
# ===========================

def evaluate_results(results_path: str, output_path: str = None, detailed_csv: str = None, plot_dir: str = None):
    print(f"[INFO] Caricamento risultati da {results_path}...")
    results = load_results(results_path)

    print(f"[INFO] Calcolo metriche biometriche...")
    scores, labels, rows = compute_labels(results)
    metrics = compute_metrics(scores, labels)

    if detailed_csv:
        save_detailed_results(rows, detailed_csv)
    if output_path:
        save_metrics(metrics, output_path)
    if plot_dir:
        plot_far_frr_curve(scores, labels, plot_dir)

    print("\n=== RISULTATI EVALUAZIONE ===")
    for k, v in metrics.items():
        print(f"{k:12s}: {v}")

    return metrics

# ===========================
# CLI
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Valutazione prestazioni biometriche con FAR-FRR curve")
    parser.add_argument("--results", required=True, help="Path al file JSON/CSV dei risultati di matching")
    parser.add_argument("--output", default="metrics/biometric_metrics.json", help="Path output per le metriche")
    parser.add_argument("--csv", default=None, help="Path CSV dettagliato con punteggi e label")
    parser.add_argument("--plot", default=None, help="Directory in cui salvare la FAR-FRR curve")

    args = parser.parse_args()
    evaluate_results(args.results, args.output, args.csv, args.plot)
