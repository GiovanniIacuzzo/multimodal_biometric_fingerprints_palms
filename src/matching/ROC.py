import numpy as np
import matplotlib.pyplot as plt
import logging
from colorama import Fore, Style
from src.matching.utils import console_step

def plot_roc(genuine_scores, impostor_scores, save_path="ROC_curve.png", n_thresholds=400):
    console_step("Generazione ROC Curve")

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # ------------------------------------------------------------
    # Calcolo ROC
    # ------------------------------------------------------------
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    tpr_list = np.zeros_like(thresholds)
    fpr_list = np.zeros_like(thresholds)

    for i, thr in enumerate(thresholds):
        tpr_list[i] = np.mean(genuine_scores >= thr)   # True Positive Rate
        fpr_list[i] = np.mean(impostor_scores >= thr)  # False Positive Rate

    # AUC con metodo trapezoidale
    auc = np.trapz(tpr_list, fpr_list)

    # ------------------------------------------------------------
    # Calcolo EER — Equal Error Rate
    # ------------------------------------------------------------
    fnr_list = 1 - tpr_list
    abs_diff = np.abs(fnr_list - fpr_list)
    eer_idx = np.argmin(abs_diff)

    eer = (fnr_list[eer_idx] + fpr_list[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    logging.info(f"EER calcolato: {eer:.4f} (threshold = {eer_threshold:.4f})")

    # ------------------------------------------------------------
    # Plot ROC migliorato
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 7), dpi=120)

    # Curva ROC
    plt.plot(
        fpr_list,
        tpr_list,
        linewidth=2.2,
        label=f"ROC Curve — AUC={auc:.4f}",
    )

    # Linea random classifier
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.2)

    # Punto EER
    plt.scatter(
        fpr_list[eer_idx],
        tpr_list[eer_idx],
        s=65,
        color='red',
        label=f"EER = {eer:.4f}",
        zorder=5
    )

    # Setting grafico
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    # Salvataggio
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"{Fore.GREEN}ROC salvata in {save_path}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}EER = {eer:.4f} (threshold = {eer_threshold:.4f}){Style.RESET_ALL}")

    logging.info(f"ROC curve salvata in {save_path} con AUC={auc:.4f} e EER={eer:.4f}")

    return fpr_list, tpr_list, auc, eer, eer_threshold
