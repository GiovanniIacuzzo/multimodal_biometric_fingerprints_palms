import matplotlib.pyplot as plt
import numpy as np

def plot_roc(thresholds, far_values, frr_values, title="ROC (FAR vs FRR)"):

    thresholds = np.array(thresholds)
    far = np.array(far_values)
    frr = np.array(frr_values)

    # Ordinare i punti per FAR crescente
    order = np.argsort(far)
    far = far[order]
    frr = frr[order]

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(far, frr, marker='o', linewidth=2)
    plt.xlabel("FAR (False Accept Rate)")
    plt.ylabel("FRR (False Reject Rate)")
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    thresholds = np.linspace(0, 1, 50)
    frr_values = [
        1.000, 0.986, 0.804, 0.446, 0.250, 0.169, 0.128, 0.108, 0.095, 0.088,
        0.061, 0.047, 0.047, 0.027, 0.014, 0.007, 0.007, 0.007, 0.007, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
    ]

    far_values = [
        0.001, 0.002, 0.219, 0.738, 0.941, 0.987, 0.998, 0.999, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
    ]

    plot_roc(thresholds, far_values=far_values, frr_values=frr_values, title="ROC (FAR vs FRR)")

    print("Curva ROC generata con successo...")