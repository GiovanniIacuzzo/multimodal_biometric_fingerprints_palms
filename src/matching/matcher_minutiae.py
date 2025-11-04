import numpy as np
import json
import os
from scipy.spatial import KDTree
import cv2

# ============================================================
#  Utility
# ============================================================
TYPE_MAPPING = {"ending": 0, "bifurcation": 1}

def _as_array(minutiae):
    if len(minutiae) == 0:
        return np.empty((0, 5), dtype=np.float32)

    if isinstance(minutiae[0], dict):
        keys = ["x", "y", "type", "orientation", "quality"]
        arr = np.array([[float(m.get(k, 0.0)) if k not in ["type","quality"] else m.get(k, 0.0) for k in keys] for m in minutiae], dtype=np.float32)
    else:
        arr = np.array(minutiae, dtype=np.float32)

    if arr.shape[1] < 5:
        pad = np.zeros((arr.shape[0], 5 - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])

    return arr

def minutiae_similarity(template, probe, dist_thresh=20.0, orient_thresh=np.pi/4):
    """
    Calcola la similarità geometrica tra due set di minutiae con verifica globale.
    """
    template = _as_array(template)
    probe = _as_array(probe)
    if len(template) == 0 or len(probe) == 0:
        return 0.0

    t_coords, p_aligned, M = align_minutiae(template, probe)
    tree = KDTree(t_coords)
    matched = []

    for i, p in enumerate(p_aligned):
        dist, j = tree.query(p)
        if dist < dist_thresh:
            ori_diff = abs((template[j, 3] - probe[i, 3] + np.pi) % (2 * np.pi) - np.pi)
            if ori_diff < orient_thresh and template[j, 2] == probe[i, 2]:
                matched.append((i, j))

    if not matched:
        return 0.0

    # === Verifica coerenza geometrica globale ===
    src_pts = np.float32([probe[i, :2] for i, _ in matched])
    dst_pts = np.float32([template[j, :2] for _, j in matched])
    if len(src_pts) >= 3:
        M2, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if inliers is not None:
            n_inliers = np.sum(inliers)
            score_geom = n_inliers / len(matched)
        else:
            score_geom = 0.0
    else:
        score_geom = 0.0

    # === Calcolo punteggio finale ===
    score_quality = np.mean([template[j, 4] * probe[i, 4] for i, j in matched])
    score = 0.5 * score_quality + 0.5 * score_geom
    return float(np.clip(score, 0.0, 1.0))

def load_minutiae(path: str):
    """
    Carica minutiae da un file JSON in modo robusto.
    Ritorna una lista di dict con x, y, type, orientation, quality.
    """
    if not os.path.exists(path):
        print(f"⚠️ File minutiae non trovato: {path}")
        return []

    try:
        with open(path, "r") as f:
            minutiae = json.load(f)
    except Exception as e:
        print(f"⚠️ Errore caricamento JSON {path}: {e}")
        return []

    if not isinstance(minutiae, list):
        print(f"⚠️ Formato JSON inatteso in {path}")
        return []

    for m in minutiae:
        m["x"] = float(m.get("x", 0.0))
        m["y"] = float(m.get("y", 0.0))
        # Trasforma type in numero se è stringa
        t = m.get("type", 0)
        if isinstance(t, str):
            m["type"] = TYPE_MAPPING.get(t.lower(), 0)
        else:
            m["type"] = int(t)
        m["orientation"] = float(m.get("orientation", 0.0))
        m["quality"] = float(m.get("quality", 1.0))
    return minutiae

def save_minutiae(path: str, minutiae):
    """Salva i minutiae in JSON."""
    with open(path, "w") as f:
        json.dump(minutiae, f, indent=2)

# ============================================================
#  Allineamento (traslazione + rotazione)
# ============================================================
def align_minutiae(template, probe, max_trials=500, dist_thresh=25.0):
    """
    Allinea le minutie del probe rispetto al template stimando una trasformazione affine robusta.
    Usa cv2.estimateAffinePartial2D con RANSAC.
    """
    t_coords, p_coords = template[:, :2], probe[:, :2]
    if len(t_coords) < 3 or len(p_coords) < 3:
        return t_coords, p_coords, np.eye(2, 3, dtype=np.float32)

    # Primo matching approssimato basato sulla vicinanza
    tree = KDTree(t_coords)
    dists, idx = tree.query(p_coords, k=1)
    good_mask = dists < dist_thresh

    src_pts = p_coords[good_mask].astype(np.float32)
    dst_pts = t_coords[idx[good_mask]].astype(np.float32)

    if len(src_pts) < 3:
        # Troppo pochi punti per stimare la trasformazione
        return t_coords, p_coords, np.eye(2, 3, dtype=np.float32)

    # Stima della trasformazione affine con RANSAC
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=max_trials,
        confidence=0.99
    )

    if M is None:
        print("⚠️ Impossibile stimare trasformazione affine.")
        return t_coords, p_coords, np.eye(2, 3, dtype=np.float32)

    # Applica la trasformazione alle minuzie del probe
    ones = np.ones((len(p_coords), 1), dtype=np.float32)
    p_h = np.hstack([p_coords, ones])
    p_aligned = (M @ p_h.T).T

    return t_coords, p_aligned, M


# ============================================================
# Test standalone + modifica JSON
# ============================================================
if __name__ == "__main__":
    file_path = "punti.json"

    # Carica minutiae dal JSON
    minutiae_data = load_minutiae(file_path)
    print(f"⚡ Minutiae caricati: {len(minutiae_data)}")

    # Aggiungi un nuovo punto di esempio
    nuovo_punto = {"x": 600.0, "y": 75.0, "type": 0, "orientation": 0.0, "quality": 1.0}
    minutiae_data.append(nuovo_punto)
    save_minutiae(file_path, minutiae_data)
    print("✅ Nuovo punto aggiunto e salvato nel JSON.")

    # Test similarità (demo tra due sottoliste)
    if len(minutiae_data) >= 2:
        score = minutiae_similarity(minutiae_data[:3], minutiae_data[1:4])
        print(f"DEBUG: Similarità geometrica: {score:.3f}")
