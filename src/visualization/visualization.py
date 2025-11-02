"""
visualization.py
----------------
Visualizzazione della pipeline fingerprint: overlay minutiae, ROC, CMC, score fusion.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================
# CONFIGURAZIONE
# ==========================
OUTPUT_DIR = "data/results/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 1. OVERLAY MINUTIAE
# ==========================
def overlay_minutiae(image_path, minutiae_json, save_path):
    """
    Sovrappone minutiae su immagine fingerprint.
    Terminazioni = rosso, Biforcazioni = blu.
    """
    img = cv2.imread(image_path)
    with open(minutiae_json) as f:
        minutiae = json.load(f)

    for m in minutiae:
        color = (0, 0, 255) if m["type"] == "ending" else (255, 0, 0)
        cv2.circle(img, (m["x"], m["y"]), 3, color, -1)

    cv2.imwrite(save_path, img)
    print(f"📌 Overlay salvato in: {save_path}")

# ==========================
# 2. PLOT ROC
# ==========================
def plot_roc(roc_csv, save_path):
    df = pd.read_csv(roc_csv)
    plt.figure()
    plt.plot(df["FPR"], df["TPR"], color="blue", lw=2)
    plt.plot([0,1], [0,1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📈 ROC plot salvato in: {save_path}")

# ==========================
# 3. PLOT CMC
# ==========================
def plot_cmc(cmc_csv, save_path):
    df = pd.read_csv(cmc_csv)
    plt.figure()
    plt.plot(df["Rank"], df["Identification Rate"], marker='o', color="green")
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.title("CMC Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 CMC plot salvato in: {save_path}")

# ==========================
# 4. ESEMPIO USO
# ==========================
if __name__ == "__main__":
    # Overlay minutiae esempio
    overlay_minutiae(
        "data/processed/001_1_1/enhanced.png",
        "data/features/minutiae/001_1_1_minutiae.json",
        os.path.join(OUTPUT_DIR, "001_1_1_minutiae_overlay.png")
    )

    # ROC plot
    plot_roc("data/results/evaluation/roc_data.csv",
             os.path.join(OUTPUT_DIR, "roc_curve.png"))

    # CMC plot
    plot_cmc("data/results/evaluation/cmc_data.csv",
             os.path.join(OUTPUT_DIR, "cmc_curve.png"))
