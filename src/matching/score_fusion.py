"""
score_fusion.py
--------------------------
Fusione adattiva dei punteggi minutiae + descriptor.
"""

import numpy as np


def adaptive_fusion(score_min, score_desc, eps=1e-6):
    """
    Fusione dinamica:
     - Se uno dei due score è basso, l'altro prevale.
     - Se entrambi sono alti, aumenta la confidenza.
    """
    # normalizzazione
    s_m = np.clip(score_min, 0.0, 1.0)
    s_d = np.clip(score_desc, 0.0, 1.0)

    # pesatura adattiva (favorisce quello più “stabile”)
    weight_m = s_m / (s_m + s_d + eps)
    weight_d = 1 - weight_m

    fused = weight_m * s_m + weight_d * s_d
    synergy = np.sqrt(s_m * s_d)  # enfatizza coerenza tra i due
    return float(np.clip(0.7 * fused + 0.3 * synergy, 0.0, 1.0))
