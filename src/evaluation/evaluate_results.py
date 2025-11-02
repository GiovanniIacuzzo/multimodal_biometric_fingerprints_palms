# src/evaluation/evaluate_results.py
"""
evaluate_results.py
-------------------
Calcolo metriche di performance biometriche: ROC, EER, CMC.
Può essere eseguito da run_pipeline.py o standalone.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_evaluation(scores_csv="data/results/scores.csv", output_dir="data/results/evaluation"):
    """
    Calcola ROC, AUC, EER e CMC a partire da un file CSV con colonne:
        probe_id, template_id, score_final, match (1=stesso soggetto, 0=diverso)
    e salva:
        - roc_curve.png
        - cmc_curve.png
        - roc_data.csv
        - cmc_data.csv
    """
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(f"⚠️ File dei risultati non trovato: {scores_csv}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(scores_csv)

    # Controlli base
    required_cols = {"probe_id", "template_id", "score_final", "match"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"⚠️ Il file CSV deve contenere le colonne: {required_cols}")

    y_true = df["match"].values
    y_score = df["score_final"].values

    # ==========================
    # ROC & AUC
    # ==========================
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # ==========================
    # EER (Equal Error Rate)
    # ==========================
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_idx]
    eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"🔹 EER: {eer_value:.3f} at threshold {eer_threshold:.3f}")

    # ==========================
    # CMC Curve
    # ==========================
    unique_probes = df["probe_id"].unique()
    ranks = []
    for probe in unique_probes:
        probe_df = df[df["probe_id"] == probe].sort_values("score_final", ascending=False)
        # Se non ci sono match positivi, salta
        if probe_df["match"].sum() == 0:
            continue
        correct_match_idx = probe_df[probe_df["match"] == 1].index[0]
        rank = probe_df.index.get_loc(correct_match_idx) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    cmc_values = [np.mean(ranks <= r) for r in range(1, len(df["template_id"].unique()) + 1)]

    plt.figure()
    plt.plot(range(1, len(cmc_values) + 1), cmc_values, marker='o', color="green")
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.title("CMC Curve")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cmc_curve.png"))
    plt.close()

    # ==========================
    # SAVE REPORTS
    # ==========================
    report_df = pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Thresholds": thresholds
    })
    report_df.to_csv(os.path.join(output_dir, "roc_data.csv"), index=False)

    cmc_df = pd.DataFrame({"Rank": range(1, len(cmc_values) + 1), "Identification Rate": cmc_values})
    cmc_df.to_csv(os.path.join(output_dir, "cmc_data.csv"), index=False)

    print(f"✅ Evaluation report saved in: {output_dir}")
    print(f"📈 AUC = {roc_auc:.3f} | EER = {eer_value:.3f}")

    return {
        "AUC": roc_auc,
        "EER": eer_value,
        "EER_threshold": eer_threshold,
        "ROC_data": report_df,
        "CMC_data": cmc_df,
    }


def main():
    """Funzione principale da richiamare in run_pipeline."""
    try:
        compute_evaluation()
    except Exception as e:
        print(f"❌ Errore durante l'evaluation: {e}")


if __name__ == "__main__":
    main()
