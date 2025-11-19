from colorama import Fore, Style
import math
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