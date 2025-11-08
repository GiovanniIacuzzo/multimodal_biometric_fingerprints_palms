import os
import json
import logging
from typing import List, Dict, Optional

import numpy as np
import cv2
from src.features.post_processing import postprocess_minutiae

# ------------------------------------------------------------
# Configurazione logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------------------------------------------------------
# Thinning (scheletrizzazione)
# ------------------------------------------------------------
def thin_skeleton(img: np.ndarray) -> np.ndarray:
    """
    Applica un'operazione di thinning sull'immagine fornita.

    Parameters
    ----------
    img : np.ndarray
        Immagine binaria o in scala di grigi (0–255).

    Returns
    -------
    np.ndarray
        Immagine binaria (0/1) corrispondente allo scheletro dell'impronta.
    """
    if img is None:
        return np.zeros_like(img, dtype=np.uint8)

    # Normalizza a 0/255
    img_u8 = (img > 0).astype(np.uint8) * 255 if img.dtype != np.uint8 else img.copy()

    try:
        # Usa thinning nativo OpenCV se disponibile
        thin = cv2.ximgproc.thinning(img_u8)
        return (thin > 0).astype(np.uint8)
    except Exception:
        # Fallback: approssimazione morfologica
        binar = (img_u8 > 127).astype(np.uint8)
        prev = np.zeros_like(binar)
        sk = binar.copy()
        for _ in range(100):
            eroded = cv2.erode(sk, np.ones((3, 3), np.uint8))
            opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
            temp = sk - opened
            sk = eroded.copy()
            if np.array_equal(sk, prev):
                break
            prev = sk.copy()
        return (sk > 0).astype(np.uint8)


# ------------------------------------------------------------
# Estrazione delle minutiae
# ------------------------------------------------------------
def extract_minutiae_from_skeleton(skel: np.ndarray) -> List[Dict]:
    """
    Estrae minutiae (terminazioni e biforcazioni) da un'immagine scheletrizzata.

    Parameters
    ----------
    skel : np.ndarray
        Immagine scheletrizzata in scala di grigi o binaria.

    Returns
    -------
    List[Dict]
        Lista di minutiae con chiavi: ``x``, ``y``, ``type``.
    """
    if skel is None:
        return []

    # Converti in grayscale se necessario
    sk_gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY) if skel.ndim == 3 else skel.copy()

    # Selezione automatica polarità (ridges bianche o nere)
    bin_pos = (sk_gray > 127).astype(np.uint8)
    bin_neg = (sk_gray <= 127).astype(np.uint8)
    count_pos, count_neg = bin_pos.sum(), bin_neg.sum()

    if count_pos == 0 and count_neg == 0:
        return []

    sk = bin_pos if count_pos <= count_neg else bin_neg
    chosen = ">127" if count_pos <= count_neg else "<=127"
    logging.info(f"Binarizzazione scelta: {chosen} (pixel ON = {min(count_pos, count_neg)})")

    # Scheletrizzazione
    sk_thin = thin_skeleton((sk * 255).astype(np.uint8))
    if sk_thin.sum() == 0:
        logging.warning("Thinning ha prodotto immagine vuota, uso binarizzazione originale.")
        sk_thin = sk

    h, w = sk_thin.shape
    minutiae: List[Dict] = []

    # Scansione 8-neighborhood per Crossing Number
    ys, xs = np.nonzero(sk_thin)
    for y, x in zip(ys, xs):
        if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
            continue

        P = [
            sk_thin[y, x + 1], sk_thin[y - 1, x + 1],
            sk_thin[y - 1, x], sk_thin[y - 1, x - 1],
            sk_thin[y, x - 1], sk_thin[y + 1, x - 1],
            sk_thin[y + 1, x], sk_thin[y + 1, x + 1],
        ]
        P_int = [int(v) for v in P]
        CN = sum(abs(P_int[i] - P_int[(i + 1) % 8]) for i in range(8)) // 2

        if CN == 1:
            minutiae.append({"x": int(x), "y": int(y), "type": "ending"})
        elif CN == 3:
            minutiae.append({"x": int(x), "y": int(y), "type": "bifurcation"})

    logging.info(f"Minutiae grezze estratte: {len(minutiae)}")
    return minutiae


# ------------------------------------------------------------
# Elaborazione singolo campione
# ------------------------------------------------------------
def process_sample(debug_dir: str, output_dir: str, params: Optional[Dict] = None) -> None:
    """
    Esegue l'intero processo di estrazione e post-processing su un singolo campione.

    Parameters
    ----------
    debug_dir : str
        Directory contenente le immagini intermedie di debug.
    output_dir : str
        Directory dove salvare i risultati.
    params : dict, optional
        Parametri per la fase di post-processing.
    """
    sample_name = os.path.basename(debug_dir.rstrip("/\\"))
    logging.info(f"Elaborazione campione: {sample_name}")

    skel_path = os.path.join(debug_dir, f"{sample_name}_skeleton.png")
    gray_path = os.path.join(debug_dir, f"{sample_name}_segmented.png")

    if not os.path.exists(skel_path):
        logging.error(f"File skeleton non trovato: {skel_path}")
        return

    if not os.path.exists(gray_path):
        logging.warning("File segmentato non trovato, uso skeleton come fallback.")
        gray_path = skel_path

    skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

    raw_minutiae = extract_minutiae_from_skeleton(skel)
    refined = postprocess_minutiae(raw_minutiae, skel, gray, params)

    logging.info(f"Minutiae finali dopo post-processing: {len(refined)}")

    os.makedirs(output_dir, exist_ok=True)
    img_out = os.path.join(output_dir, f"{sample_name}_minutiae_postprocessed.png")
    json_out = os.path.join(output_dir, f"{sample_name}_minutiae.json")

    # Visualizzazione dei risultati
    vis = cv2.cvtColor((skel > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for m in refined:
        color = (0, 0, 255) if m.get("type") == "ending" else (0, 255, 0)
        cv2.circle(vis, (int(m["x"]), int(m["y"])), 3, color, -1)
    cv2.imwrite(img_out, vis)

    # Salvataggio in JSON
    with open(json_out, "w") as f:
        json.dump(refined, f, indent=2)

    logging.info(f"Risultati salvati in: {output_dir}")


# ------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------
def main(input_base: Optional[str] = None, output_base: Optional[str] = None):
    """
    Esegue l'elaborazione di tutte le impronte in una directory di input.

    Parameters
    ----------
    input_base : str, optional
        Percorso base di input.
    output_base : str, optional
        Percorso base di output.
    """
    input_base = input_base or os.path.join("data", "processed", "debug")
    output_base = output_base or os.path.join("data", "features", "minutiae")

    if not os.path.exists(input_base):
        raise FileNotFoundError(f"Cartella non trovata: {input_base}")

    os.makedirs(output_base, exist_ok=True)
    sample_dirs = [os.path.join(input_base, d) for d in os.listdir(input_base)
                   if os.path.isdir(os.path.join(input_base, d))]

    if not sample_dirs:
        logging.warning("Nessuna sottocartella trovata.")
        return

    logging.info(f"Trovate {len(sample_dirs)} impronte da elaborare.")
    for debug_dir in sample_dirs:
        sample_name = os.path.basename(debug_dir.rstrip('/\\'))
        try:
            process_sample(debug_dir, os.path.join(output_base, sample_name))
        except Exception as e:
            logging.error(f"Errore durante l'elaborazione di {sample_name}: {e}")


if __name__ == "__main__":
    main()
