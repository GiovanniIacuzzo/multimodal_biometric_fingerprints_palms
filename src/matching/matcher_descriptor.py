"""
matcher_descriptor.py
---------------------
Matching tra fingerprint basato su feature descriptors (handcrafted o deep).
"""

import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def load_descriptor(path: str):
    """Carica vettore o matrice di feature (.npy) e restituisce 1D normalizzato."""
    desc = np.load(path)
    if desc.size == 0:
        print(f"⚠️ Descriptor vuoto in {path}")
        return np.zeros((1,), dtype=float)
    if desc.ndim > 1:
        # Pooling: media lungo le righe
        desc = desc.mean(axis=0)
    norm = np.linalg.norm(desc)
    if norm == 0:
        return np.zeros_like(desc)
    return desc / norm


def descriptor_similarity(template_path: str, probe_path: str) -> float:
    """
    Calcola similarità tra due fingerprint descriptors (cosine similarity).
    Restituisce 0 se uno dei due è vuoto.
    """
    t = load_descriptor(template_path)
    p = load_descriptor(probe_path)
    if t.size == 0 or p.size == 0:
        return 0.0
    sim = cosine_similarity([t], [p])[0, 0]
    return float(sim)


def batch_match(probe_path: str, gallery_dir: str):
    """
    Confronta un probe con tutti i template in una directory (1:N matching).
    """
    scores = {}
    probe_vec = load_descriptor(probe_path)
    for file in os.listdir(gallery_dir):
        if file.endswith(".npy"):
            template_path = os.path.join(gallery_dir, file)
            template_vec = load_descriptor(template_path)
            sim = cosine_similarity([probe_vec], [template_vec])[0, 0]
            scores[file] = sim
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    probe = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/descriptors_handcrafted/1_1_1_features.npy"
    template = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/descriptors_handcrafted/1_1_2_features.npy"
    score = descriptor_similarity(template, probe)
    print(f"📊 Descriptor similarity score: {score:.3f}")
