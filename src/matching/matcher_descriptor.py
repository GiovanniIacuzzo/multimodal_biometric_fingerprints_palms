import numpy as np
import os

# ============================================================
#  Caricamento descriptor in modo robusto
# ============================================================
def safe_load_descriptor(path: str) -> np.ndarray:
    """
    Caricamento sicuro descriptor .npy.
    Ritorna un array Nx4 in float32, vuoto se file mancante o non valido.
    """
    if not os.path.exists(path):
        print(f"⚠️ Descriptor non trovato: {path}")
        return np.empty((0, 4), dtype=np.float32)

    try:
        desc = np.load(path)
        if desc.ndim != 2 or desc.shape[1] < 3:
            print(f"⚠️ Descriptor formato non valido (shape {desc.shape}): {path}")
            return np.empty((0, 4), dtype=np.float32)
        # Assicuriamoci che abbia almeno 4 colonne
        if desc.shape[1] < 4:
            pad = np.zeros((desc.shape[0], 4 - desc.shape[1]), dtype=np.float32)
            desc = np.hstack([desc, pad])
        return desc.astype(np.float32)
    except Exception as e:
        print(f"⚠️ Errore caricamento descriptor {path}: {e}")
        return np.empty((0, 4), dtype=np.float32)

# ============================================================
#  Similarità descriptor
# ============================================================
def descriptor_similarity(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Calcola la similarità tra due array descriptor.
    Restituisce punteggio tra 0 e 1.
    """
    if desc1.size == 0 or desc2.size == 0:
        return 0.0

    try:
        coords1, coords2 = desc1[:, :2], desc2[:, :2]
        types1, types2 = desc1[:, 2], desc2[:, 2]

        # distanza Euclidea
        dists = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)

        # penalità basata sul tipo
        type_diff = np.abs(types1[:, None] - types2[None, :])
        penalty = np.clip(1.0 - 0.5 * type_diff, 0.0, 1.0)

        # pesi esponenziali sulle distanze
        weights = np.exp(-dists / 20.0) * penalty

        sim = np.mean(np.max(weights, axis=1))
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        print(f"❌ Errore nel calcolo similarità descriptor: {e}")
        return 0.0

# ============================================================
# Test standalone rapido
# ============================================================
if __name__ == "__main__":
    desc1 = np.array([[10, 20, 0, 0.5],
                      [30, 40, 1, 0.3]], dtype=np.float32)
    desc2 = np.array([[12, 22, 0, 0.5],
                      [29, 41, 1, 0.3]], dtype=np.float32)

    score = descriptor_similarity(desc1, desc2)
    print(f"DEBUG: Similarità descriptor: {score:.3f}")
