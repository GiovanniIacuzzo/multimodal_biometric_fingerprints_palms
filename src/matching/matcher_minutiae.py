"""
matcher_minutiae.py
-------------------
Matching geometrico tra due fingerprint basato su minutiae (terminazioni/biforcazioni).
"""

import json
import numpy as np
from scipy.spatial import distance_matrix


def load_minutiae(json_path: str):
    """Carica minutiae da file JSON."""
    with open(json_path) as f:
        return json.load(f)


def align_minutiae(template, probe):
    """
    Esegue un semplice allineamento traslazionale basato sui centroidi.
    Restituisce array vuoti se non ci sono minutiae.
    """
    t_coords = np.array([[m["x"], m["y"]] for m in template]) if template else np.empty((0, 2))
    p_coords = np.array([[m["x"], m["y"]] for m in probe]) if probe else np.empty((0, 2))

    if t_coords.size == 0 or p_coords.size == 0:
        return t_coords, p_coords  # array vuoti

    t_center = t_coords.mean(axis=0)
    p_center = p_coords.mean(axis=0)
    p_aligned = p_coords - p_center + t_center
    return t_coords, p_aligned


def minutiae_similarity(template, probe, dist_thresh=15, orient_thresh=np.pi/6):
    """
    Calcola un punteggio di similarità tra due set di minutiae.
    Restituisce 0 se uno dei due set è vuoto.
    """
    if not template or not probe:
        return 0.0

    t_coords, p_coords = align_minutiae(template, probe)

    if t_coords.size == 0 or p_coords.size == 0:
        return 0.0

    # Calcola la matrice delle distanze
    dist = distance_matrix(t_coords, p_coords)

    matches = 0
    for i in range(len(template)):
        min_idx = np.argmin(dist[i])
        if dist[i, min_idx] < dist_thresh:
            ori_diff = abs(template[i]["orientation"] - probe[min_idx]["orientation"])
            if ori_diff < orient_thresh:
                matches += 1

    score = matches / max(len(template), len(probe))
    return score


if __name__ == "__main__":
    template_path = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/features/minutiae/1_1_1_minutiae.json"
    probe_path    = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/features/minutiae/1_1_2_minutiae.json"

    template = load_minutiae(template_path)
    probe = load_minutiae(probe_path)

    score = minutiae_similarity(template, probe)
    print(f"🔐 Minutiae similarity score: {score:.3f}")
