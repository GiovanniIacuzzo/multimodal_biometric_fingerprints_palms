"""
matcher_descriptor.py
------------------------------
Matching tra fingerprint basato su feature descriptors (deep o handcrafted).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_descriptor(path: str):
    try:
        desc = np.load(path)
        if desc.ndim > 1:
            desc = desc.mean(axis=0)
        desc = desc.astype(np.float32)
        norm = np.linalg.norm(desc)
        return desc / norm if norm > 0 else np.zeros_like(desc)
    except Exception:
        return np.zeros((128,), dtype=np.float32)  # fallback dimension


def descriptor_similarity(template_path: str, probe_path: str) -> float:
    """Calcola la cosine similarity tra due fingerprint descriptors."""
    t = load_descriptor(template_path)
    p = load_descriptor(probe_path)
    if t.size == 0 or p.size == 0:
        return 0.0
    sim = cosine_similarity([t], [p])[0, 0]
    return float(max(0.0, sim))  # evita negativi
