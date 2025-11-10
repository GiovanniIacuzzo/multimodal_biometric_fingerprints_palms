import os
import json
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
from scipy.stats import norm
import warnings

# ============================================================
# 1. Caricamento risultati
# ============================================================

def load_results(results_path: str) -> Dict[Tuple[str, str], float]:
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"File risultati non trovato: {results_path}")

    results = {}
    if results_path.endswith(".json"):
        with open(results_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in raw.items():
            if isinstance(key, str) and "_vs_" in key:
                u1, u2 = key.split("_vs_")
                results[(u1, u2)] = float(value)
            elif isinstance(key, (list, tuple)) and len(key) == 2:
                results[tuple(key)] = float(value)
            else:
                print(f"[WARN] Chiave JSON non riconosciuta: {key}")
    elif results_path.endswith(".csv"):
        with open(results_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[(row["user1"], row["user2"])] = float(row["score"])
    else:
        raise ValueError("Formato non supportato (solo .json o .csv)")

    print(f"[INFO] Caricati {len(results)} risultati da {results_path}")
    return results


# ============================================================
# 2. Estrazione ID e labeling
# ============================================================

def extract_id(filename: str) -> str:
    """
    Estrae l'ID biometrico da un nome file o ID generico.
    Gestisce formati:
        - '100_1_1_minutiae_postprocessed.jpg'
        - 'data/.../100_1_1_binary.jpg'
        - '100_1_1'
        - '1'
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)

    # Cerca pattern tipo "100_1_1" o "001_2_05"
    match = re.match(r"(\d+)_\d+_\d+", name)
    if match:
        return match.group(1)

    # Solo numerico
    if name.isdigit():
        return name

    # Tenta di estrarre la parte numerica iniziale
    match = re.match(r"(\d+)", name)
    if match:
        return match.group(1)

    raise ValueError(f"Formato nome file non valido o non numerico: {filename}")


def compute_labels(results: Dict[Tuple[str, str], float]):
    scores, labels, rows = [], [], []
    for (u1, u2), score in results.items():
        try:
            id1, id2 = extract_id(u1), extract_id(u2)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        is_genuine = int(id1 == id2)
        scores.append(score)
        labels.append(is_genuine)
        rows.append((u1, u2, score, is_genuine))

    scores, labels = np.array(scores), np.array(labels)
    genuine_count = int(np.sum(labels))
    impostor_count = len(labels) - genuine_count
    print(f"[INFO] Coppie totali: {len(labels)} | Genuine: {genuine_count} | Impostor: {impostor_count}")

    return scores, labels, rows


# ============================================================
# 3. Calcolo metriche biometriche robuste
# ============================================================

def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels)) < 2:
        print("[WARN] Solo una classe trovata nei dati, metriche limitate.")
        return {
            "EER": None,
            "Threshold_EER": None,
            "AUC": None,
            "Accuracy": float(np.mean(labels == 0)),
            "Precision": None,
            "Recall": None,
            "F1-score": None,
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0
        }

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_value = auc(fpr, tpr)
    fnr = 1 - tpr
    idx_eer = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    threshold_eer = thresholds[idx_eer]

    preds = (scores >= threshold_eer).astype(int)
    accuracy = np.mean(preds == labels)

    # Soppressione warning sklearn su precision/recall
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "EER": round(float(eer), 4),
        "Threshold_EER": round(float(threshold_eer), 4),
        "AUC": round(float(auc_value), 4),
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-score": round(float(f1), 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }

    return metrics


# ============================================================
# 4. Salvataggio risultati
# ============================================================

def save_metrics(metrics: Dict[str, float], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metriche salvate in {output_path}")


def save_detailed_results(rows: List[Tuple[str, str, float, int]], output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user1", "user2", "score", "label"])
        writer.writerows(rows)
    print(f"[INFO] Risultati dettagliati salvati in {output_csv}")


# ============================================================
# 5. Curve biometriche
# ============================================================

def plot_far_frr_curve(scores: np.ndarray, labels: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    thresholds = np.sort(scores)
    total_genuine = np.sum(labels)
    total_impostor = len(labels) - total_genuine
    FARs, FRRs = [], []

    for t in thresholds:
        FA = np.sum((scores >= t) & (labels == 0))
        FR = np.sum((scores < t) & (labels == 1))
        FAR = FA / total_impostor if total_impostor > 0 else 0
        FRR = FR / total_genuine if total_genuine > 0 else 0
        FARs.append(FAR)
        FRRs.append(FRR)

    plt.figure(figsize=(6, 6))
    plt.plot(FARs, FRRs, lw=2, color='blue', label='FAR vs FRR')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='EER line')
    plt.xlabel("FAR (False Acceptance Rate)")
    plt.ylabel("FRR (False Rejection Rate)")
    plt.title("FAR vs FRR Curve")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "far_frr_curve.png"))
    plt.close()
    print(f"[INFO] FAR-FRR curve salvata in {output_dir}")


def plot_roc_det(scores: np.ndarray, labels: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr

    # ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # DET
    plt.figure(figsize=(6, 6))
    plt.plot(norm.ppf(fpr), norm.ppf(fnr), lw=2)
    plt.xlabel("FAR [norm dev]")
    plt.ylabel("FRR [norm dev]")
    plt.title("DET Curve")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "det_curve.png"))
    plt.close()
    print(f"[INFO] ROC e DET curve salvate in {output_dir}")


# ============================================================
# 6. Funzione principale
# ============================================================

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
        plot_roc_det(scores, labels, plot_dir)

    print("\n=== RISULTATI EVALUAZIONE ===")
    for k, v in metrics.items():
        print(f"{k:15s}: {v}")

    return metrics


# ============================================================
# 7. CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Valutazione prestazioni biometriche (PolyU HRF DBII)")
    parser.add_argument("--results", required=True, help="Path al file JSON/CSV dei risultati di matching")
    parser.add_argument("--output", default="metrics/biometric_metrics.json", help="Path di output JSON")
    parser.add_argument("--csv", default="metrics/detailed_results.csv", help="Path CSV dettagliato")
    parser.add_argument("--plot", default="metrics/plots", help="Directory per le curve biometriche")
    args = parser.parse_args()

    evaluate_results(args.results, args.output, args.csv, args.plot)
