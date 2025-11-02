"""
score_fusion.py
----------------
Fusione dei punteggi di similarità (minutiae + descriptor) in un unico score normalizzato.
"""

def normalize_score(score, method="minmax", min_val=0.0, max_val=1.0):
    """
    Normalizza uno score tra 0 e 1 (default).
    """
    if method == "minmax":
        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))
    elif method == "sigmoid":
        import math
        return 1 / (1 + math.exp(-score))
    return score


def fuse_scores(score_minutiae, score_descriptor, weight_minutiae=0.5):
    """
    Fusione pesata dei due score.
    weight_minutiae = importanza relativa del punteggio minutiae
    """
    weight_descriptor = 1 - weight_minutiae
    # Normalizzazione preliminare
    s_m = normalize_score(score_minutiae)
    s_d = normalize_score(score_descriptor)
    # Fusione pesata
    s_final = weight_minutiae * s_m + weight_descriptor * s_d
    return s_final


if __name__ == "__main__":
    # Esempio
    score_minutiae = 0.412
    score_descriptor = 0.876
    score_final = fuse_scores(score_minutiae, score_descriptor, weight_minutiae=0.4)
    print(f"🔗 Score finale normalizzato: {score_final:.3f}")
