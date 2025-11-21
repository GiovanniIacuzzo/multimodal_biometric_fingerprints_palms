import numpy as np
import matplotlib.pyplot as plt

def plot_roc(genuine_scores, impostor_scores, title="ROC Curve", n_thr=1000):

    scores = np.concatenate([genuine_scores, impostor_scores])

    thr_min = scores.min() - 1e-6
    thr_max = scores.max() + 1e-6
    thresholds = np.linspace(thr_min, thr_max, n_thr)

    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    tpr = []
    fpr = []

    for t in thresholds:
        tpr.append(np.mean(genuine >= t))   # True Positive Rate
        fpr.append(np.mean(impostor >= t))  # False Positive Rate

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    # ---- EER ----
    diff = np.abs(fpr - (1 - tpr))
    eer_idx = diff.argmin()
    eer = fpr[eer_idx]

    # ---- AUC ----
    auc = np.trapz(tpr[::-1], fpr[::-1])

    # ---- Plot ----
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], marker='o', color='red')
    plt.title(f"{title}\nAUC={auc:.4f}  EER={eer:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()

    print("AUC:", auc)
    print("EER:", eer)
