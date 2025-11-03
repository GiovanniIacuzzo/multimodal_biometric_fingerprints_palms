import numpy as np
import json
import os
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor

# ============================================================
#  Utility
# ============================================================
TYPE_MAPPING = {"ending": 0, "bifurcation": 1}  # mappa stringhe a numeri

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
    template = _as_array(template)
    probe = _as_array(probe)
    if len(template) == 0 or len(probe) == 0:
        return 0.0

    t_coords, p_aligned = align_minutiae(template, probe)
    tree = KDTree(t_coords)
    matched_score = 0.0

    for i, p in enumerate(p_aligned):
        dist, j = tree.query(p)
        if dist < dist_thresh:
            ori_diff = abs((template[j,3] - probe[i,3] + np.pi) % (2*np.pi) - np.pi)
            if ori_diff < orient_thresh and template[j,2] == probe[i,2]:
                matched_score += template[j,4] * probe[i,4]  # peso qualità

    score = matched_score / max(len(template), len(probe))
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
def align_minutiae(template, probe, max_trials=100):
    t_coords, p_coords = template[:, :2], probe[:, :2]
    if len(t_coords) < 3 or len(p_coords) < 3:
        return t_coords, p_coords

    try:
        tree = KDTree(t_coords)
        _, idx = tree.query(p_coords, k=1)
        model_x = RANSACRegressor(min_samples=3, max_trials=max_trials)
        model_y = RANSACRegressor(min_samples=3, max_trials=max_trials)
        model_x.fit(p_coords, t_coords[idx][:, 0])
        model_y.fit(p_coords, t_coords[idx][:, 1])
        pred_x = model_x.predict(p_coords)
        pred_y = model_y.predict(p_coords)
        aligned = np.stack([pred_x, pred_y], axis=1)
        return t_coords, aligned
    except Exception as e:
        print(f"⚠️ Errore allineamento minutiae: {e}")
        return t_coords, p_coords

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
