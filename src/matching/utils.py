from colorama import Fore, Style
import math
from typing import List
import numpy as np

def console_step(title: str):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

def rotate_points(points: np.ndarray, theta: float) -> np.ndarray:
    """Ruota punti Nx2 attorno all'origine di theta radianti."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s],[s, c]])
    return points.dot(R.T)

def angle_diff(a, b):
    """Differenza angolare normale tra a e b (radianti) in [-pi, pi]."""
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

# ------------------------------------------------------------
# Report Helpers
# ------------------------------------------------------------
def report_scores(title: str, scores: List[float]):
    print("\n===========================================")
    print(f" {title}")
    print("===========================================")
    print(f"Num campioni: {len(scores)}")
    if len(scores) > 0:
        print(f"Media:  {np.mean(scores):.4f}")
        print(f"Min:    {np.min(scores):.4f}")
        print(f"Max:    {np.max(scores):.4f}")
        print(f"Std:    {np.std(scores):.4f}")
    print("===========================================\n")
def evaluate_frr_across_thresholds(genuine_scores, num_points=50):
    """
    Restituisce il vettore FRR per soglie equidistanziate.
    FRR(threshold) = percentuale di Genuine Score > threshold
    """
    print("\n===========================================")
    print(" VALORI FRR AL VARIARE DELLA SOGLIA")
    print("===========================================\n")
    print(f"{'Soglia':>8} | {'FRR':>8}")
    print("-" * 22)

    thresholds = np.linspace(0, 1, num_points)
    frr_list = []

    for t in thresholds:
        frr = np.mean(np.array(genuine_scores) > t)
        frr_list.append(frr)
        print(f"{t:8.3f} | {frr:8.3f}")

    print("\n===========================================\n")

    return thresholds, np.array(frr_list)


def evaluate_far_across_thresholds(impostor_scores, num_points=50):
    """
    Restituisce il vettore FAR per soglie equidistanziate.
    FAR(threshold) = percentuale di impostor_score <= threshold
    """
    print("\n===========================================")
    print(" VALORI FAR AL VARIARE DELLA SOGLIA")
    print("===========================================\n")
    print(f"{'Soglia':>8} | {'FAR':>8}")
    print("-" * 22)

    thresholds = np.linspace(0, 1, num_points)
    far_list = []

    for t in thresholds:
        far = np.mean(np.array(impostor_scores) <= t)
        far_list.append(far)
        print(f"{t:8.3f} | {far:8.3f}")

    print("\n===========================================\n")

    return thresholds, np.array(far_list)