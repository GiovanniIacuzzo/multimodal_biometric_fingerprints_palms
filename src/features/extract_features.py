import os
import cv2
import numpy as np
from typing import List, Dict, Optional

from src.features.utils import compute_orientation_map
from config import config as cfg

# ============================================================
# 1. Stima orientamento locale di una minutia
# ============================================================
def estimate_minutia_orientation(skel: np.ndarray, x: int, y: int, window: Optional[int] = None) -> float:
    """Stima l'orientamento di una singola minutia su immagine scheletrizzata."""
    if window is None:
        window = cfg.ORIENTATION_WINDOW if hasattr(cfg, "ORIENTATION_WINDOW") else 15

    h, w = skel.shape
    r = window // 2
    x1, x2 = max(0, x - r), min(w, x + r + 1)
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    patch = skel[y1:y2, x1:x2]

    ys, xs = np.nonzero(patch)
    if len(xs) < 10:
        return np.nan

    coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    cov = coords.T @ coords
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(main_dir[1], main_dir[0])
    return float(np.mod(angle + np.pi, np.pi) - np.pi / 2)


# ============================================================
# 2. Non-Maximum Suppression spaziale
# ============================================================
from scipy.spatial import cKDTree

def nms_min_distance(minutiae: List[Dict], min_dist: Optional[float] = None) -> List[Dict]:
    """Applica NMS spaziale per rimuovere minutiae troppo vicine."""
    if min_dist is None:
        min_dist = getattr(cfg, "MIN_DISTANCE", 6.0)

    if not minutiae:
        return []

    coords = np.array([[m["x"], m["y"]] for m in minutiae])
    qualities = np.array([m.get("quality", 1.0) for m in minutiae])
    order = np.argsort(-qualities)

    keep_mask = np.zeros(len(minutiae), dtype=bool)
    tree = cKDTree(coords)

    for i in order:
        if keep_mask[i]:
            continue
        keep_mask[i] = True
        neighbors = tree.query_ball_point(coords[i], r=min_dist)
        for j in neighbors:
            if j != i:
                keep_mask[j] = False

    return [m for i, m in enumerate(minutiae) if keep_mask[i]]


# ============================================================
# 3. Post-processing completo delle minutiae
# ============================================================
def postprocess_minutiae(
    minutiae: List[Dict],
    skel: np.ndarray,
    gray: Optional[np.ndarray] = None,
    params: Optional[Dict] = None
) -> List[Dict]:
    """
    Post-processing completo delle minutiae:
      - Filtraggio per margine, coerenza e densità
      - Stima dell’orientamento locale
      - Non-Maximum Suppression
    """
    if not minutiae or skel is None:
        return []

    params = params or {}
    qwin = params.get("quality_window", getattr(cfg, "QUALITY_WINDOW", 25))
    qth = params.get("quality_threshold", getattr(cfg, "QUALITY_THRESHOLD", 0.1))
    coh_th = params.get("coherence_threshold", getattr(cfg, "COHERENCE_THRESHOLD", 0.15))
    min_dist = params.get("min_distance", getattr(cfg, "MIN_DISTANCE", 6.0))
    ori_win = params.get("orientation_window", getattr(cfg, "ORIENTATION_WINDOW", 15))
    margin = params.get("margin", getattr(cfg, "MARGIN", 35))

    sk_bin = (skel > 0).astype(np.uint8)
    h, w = sk_bin.shape

    # Mappa di densità normalizzata
    density = cv2.blur(sk_bin.astype(np.float32), (qwin, qwin))
    density /= (density.max() + 1e-6)

    # Mappa di orientazione e coerenza
    orient, coherence = compute_orientation_map(gray if gray is not None else sk_bin)
    coherence = np.clip(coherence, 0, 1)

    enriched = []
    for m in minutiae:
        x, y = int(m["x"]), int(m["y"])

        if x < margin or x > (w - margin) or y < margin or y > (h - margin):
            continue

        local_coh = float(coherence[y, x])
        local_density = float(density[y, x])

        if local_density < qth or local_coh < coh_th:
            continue

        ang = estimate_minutia_orientation(sk_bin, x, y, ori_win)
        if np.isnan(ang):
            continue

        q = 0.6 * local_coh + 0.4 * local_density
        m.update({
            "orientation": ang,
            "quality": q,
            "coherence": local_coh
        })
        enriched.append(m)

    refined = nms_min_distance(enriched, min_dist)
    return refined


# ============================================================
# 4. MAIN: Esecuzione di test / validazione
# ============================================================
def process_sample(debug_dir: str):
    """
    Esegue il post-processing per un singolo sample contenuto in debug_dir.
    """
    sample_name = os.path.basename(debug_dir)
    print(f"\n--- Elaborazione di {sample_name} ---")

    # Percorsi immagini
    skel_path = os.path.join(debug_dir, f"{sample_name}_skeleton.png")
    gray_path = os.path.join(debug_dir, f"{sample_name}_segmented.png")
    output_path = os.path.join(debug_dir, f"{sample_name}_minutiae_postprocessed.png")

    # Verifica file richiesti
    if not os.path.exists(skel_path):
        print(f"❌ Nessuna immagine skeleton trovata per {sample_name}, salto.")
        return

    if not os.path.exists(gray_path):
        print(f"⚠️  Immagine segmentata non trovata per {sample_name}, uso skeleton come fallback.")
        gray_path = skel_path

    # Caricamento immagini
    skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

    # --- Simulazione minutiae (provvisoria, in attesa di estrazione reale) ---
    test_minutiae = [
        {"x": np.random.randint(40, skel.shape[1] - 40),
         "y": np.random.randint(40, skel.shape[0] - 40)}
        for _ in range(100)
    ]

    print(f"Minutiae iniziali: {len(test_minutiae)}")

    # --- Post-processing ---
    refined = postprocess_minutiae(test_minutiae, skel, gray)
    print(f"Minutiae finali:   {len(refined)}")

    # --- Visualizzazione del risultato ---
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for m in refined:
        x, y = int(m["x"]), int(m["y"])
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    # --- Salvataggio ---
    cv2.imwrite(output_path, vis)
    print(f"✅ Risultato salvato in: {output_path}")


def main():
    """
    Scorre automaticamente tutte le directory in data/processed/debug/
    ed esegue il post-processing per ciascuna.
    """
    base_dir = os.path.join("data", "processed", "debug")

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Cartella non trovata: {base_dir}")

    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]

    if not subdirs:
        print("Nessuna sottocartella trovata in data/processed/debug/")
        return

    for debug_dir in subdirs:
        try:
            process_sample(debug_dir)
        except Exception as e:
            print(f"⚠️ Errore durante l'elaborazione di {debug_dir}: {e}")


if __name__ == "__main__":
    main()
