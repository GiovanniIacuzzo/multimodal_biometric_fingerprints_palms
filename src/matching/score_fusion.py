"""
score_fusion.py
-----------------------------------
Fusione adattiva dei punteggi:
 - geometrico (minutiae_similarity)
 - descrittore (descriptor_similarity)

La fusione tiene conto:
 - della qualità media delle minuzie
 - della coerenza tra i due punteggi
 - di un adattamento dinamico dei pesi
"""

import numpy as np

def adaptive_fusion(score_min, score_desc, quality_min=None, eps=1e-6):
    s_m = np.clip(score_min, 0.0, 1.0)
    s_d = np.clip(score_desc, 0.0, 1.0)

    if quality_min is None:
        weight_m = s_m / (s_m + s_d + eps)
    else:
        q = np.clip(quality_min, 0.0, 1.0)
        weight_m = 0.3 + 0.7 * q
        weight_m *= s_m / (s_m + s_d + eps)

    weight_d = 1.0 - weight_m
    fused = weight_m * s_m + weight_d * s_d
    synergy = np.sqrt(s_m * s_d)
    score_final = 0.7 * fused + 0.3 * synergy
    return float(np.clip(score_final, 0.0, 1.0))

# ============================================================
#  Test standalone
# ============================================================
if __name__ == "__main__":
    tests = [
        (0.9, 0.8, 0.9),   # buona qualità, coerenza alta
        (0.4, 0.9, 0.2),   # qualità bassa, più peso al descrittore
        (0.7, 0.1, 0.8),   # minutiae forti, descriptor debole
        (0.5, 0.5, 0.5),   # equilibrio medio
    ]
    for s_m, s_d, q in tests:
        fused = adaptive_fusion(s_m, s_d, q)
        print(f"min={s_m:.2f}, desc={s_d:.2f}, q={q:.2f} → fused={fused:.3f}")
