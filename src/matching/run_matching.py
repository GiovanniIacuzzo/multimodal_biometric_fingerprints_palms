# src/matching/run_matching.py
"""
run_matching.py
---------------
Esegue matching 1:N su tutto il dataset usando:
 - matcher_minutiae.minutiae_similarity
 - matcher_descriptor.descriptor_similarity
 - score_fusion.fuse_scores

Genera: data/results/scores.csv
Colonne: probe_id, template_id, score_minutiae, score_descriptor, score_final, match
"""

import os
import csv
import glob
import traceback
import numpy as np
import pandas as pd

# tenta import config (se esiste), altrimenti fallback sui path relativi
try:
    from scripts.config import DATASET_DIR, FEATURES_DIR, RESULTS_DIR, ensure_dirs
except Exception:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET_DIR = os.path.join(BASE_DIR, "data")
    FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
    RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
    def ensure_dirs():
        os.makedirs(RESULTS_DIR, exist_ok=True)

# matcher modules
from src.matching.matcher_minutiae import load_minutiae, minutiae_similarity
from src.matching.matcher_descriptor import descriptor_similarity
from src.matching.score_fusion import fuse_scores

# default subfolders (convenzionali nella tua repo)
MINUTIAE_DIR = os.path.join(FEATURES_DIR, "minutiae")
DESC_HAND_DIR = os.path.join(FEATURES_DIR, "descriptors_handcrafted")
DESC_DEEP_DIR = os.path.join(FEATURES_DIR, "descriptors_deep")

# output CSV path (evaluate_results.py si aspetta data/results/scores.csv)
OUTPUT_SCORES_CSV = os.path.join(RESULTS_DIR, "scores.csv")

def find_descriptor_for(base_name: str):
    """
    Cerca un file descriptor corrispondente a base_name nelle cartelle note.
    Ritorna percorso file o None.
    Cerca (in ordine):
      - descriptors_deep/<base>_features.npy
      - descriptors_deep/<base>.npy
      - descriptors_handcrafted/<base>_features.npy
      - descriptors_handcrafted/<base>.npy
      - qualsiasi file .npy che inizia con base_name nei due dir
    """
    candidates = [
        os.path.join(DESC_DEEP_DIR, f"{base_name}_features.npy"),
        os.path.join(DESC_DEEP_DIR, f"{base_name}.npy"),
        os.path.join(DESC_HAND_DIR, f"{base_name}_features.npy"),
        os.path.join(DESC_HAND_DIR, f"{base_name}.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # fallback: glob
    for d in [DESC_DEEP_DIR, DESC_HAND_DIR]:
        if os.path.isdir(d):
            gl = glob.glob(os.path.join(d, f"{base_name}*.npy"))
            if gl:
                return gl[0]
    return None

def find_minutiae_for(base_name: str):
    """
    Cerca file minutiae per base_name: <base>_minutiae.json
    """
    path = os.path.join(MINUTIAE_DIR, f"{base_name}_minutiae.json")
    if os.path.exists(path):
        return path
    # fallback: glob
    if os.path.isdir(MINUTIAE_DIR):
        gl = glob.glob(os.path.join(MINUTIAE_DIR, f"{base_name}*minutiae*.json"))
        if gl:
            return gl[0]
    return None

def safe_load_minutiae(path):
    try:
        return load_minutiae(path)
    except Exception:
        return []

def compute_pair_scores(probe_base, template_base, weight_minutiae=0.5):
    """
    Restituisce (score_minutiae, score_descriptor, score_final)
    """
    # minutiae
    minutiae_probe_path = find_minutiae_for(probe_base)
    minutiae_template_path = find_minutiae_for(template_base)
    score_min = 0.0
    if minutiae_probe_path and minutiae_template_path:
        try:
            probe_min = safe_load_minutiae(minutiae_probe_path)
            templ_min = safe_load_minutiae(minutiae_template_path)
            score_min = float(minutiae_similarity(templ_min, probe_min))
        except Exception:
            score_min = 0.0

    # descriptor
    desc_probe = find_descriptor_for(probe_base)
    desc_template = find_descriptor_for(template_base)
    score_desc = 0.0
    if desc_probe and desc_template:
        try:
            score_desc = float(descriptor_similarity(desc_template, desc_probe))
        except Exception:
            score_desc = 0.0

    # fused
    score_final = fuse_scores(score_min, score_desc, weight_minutiae=weight_minutiae)
    return score_min, score_desc, score_final, minutiae_probe_path, minutiae_template_path, desc_probe, desc_template

def main(catalog_path: str = None, output_csv: str = None, weight_minutiae: float = 0.5, limit_pairs: int = None):
    """
    Esegue il matching su tutto il catalogo.
    - catalog_path: path a catalog.csv (se None cerca in DATASET_DIR/metadata/catalog.csv)
    - output_csv: dove salvare i risultati (default: OUTPUT_SCORES_CSV)
    - weight_minutiae: peso per la fusione
    - limit_pairs: opzionale, limita numero di coppie calcolate (utile per debug)
    """
    ensure_dirs()

    if catalog_path is None:
        catalog_path = os.path.join(DATASET_DIR, "metadata", "catalog.csv")
    if output_csv is None:
        output_csv = OUTPUT_SCORES_CSV

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"catalog non trovato: {catalog_path}")

    df = pd.read_csv(catalog_path)
    image_ids = df["image_id"].tolist()
    subject_map = dict(zip(df["image_id"], df["subject_id"]))

    # costruisco lista di coppie (probe, template)
    pairs = []
    for p in image_ids:
        for t in image_ids:
            pairs.append((p, t))
    # opzionale shuffle? per ora manteniamo ordine deterministico

    # se limit_pairs è impostato
    if limit_pairs:
        pairs = pairs[:limit_pairs]

    records = []
    total = len(pairs)
    print(f"🔎 Matching: {len(image_ids)} immagini -> {total} coppie")

    idx = 0
    for probe_base, template_base in pairs:
        idx += 1
        try:
            score_min, score_desc, score_final, min_p, min_t, desc_p, desc_t = compute_pair_scores(
                probe_base, template_base, weight_minutiae=weight_minutiae
            )

            match_label = 1 if str(subject_map.get(probe_base)) == str(subject_map.get(template_base)) else 0

            records.append({
                "probe_id": probe_base,
                "template_id": template_base,
                "score_minutiae": score_min,
                "score_descriptor": score_desc,
                "score_final": score_final,
                "match": match_label,
                "minutiae_probe_path": min_p or "",
                "minutiae_template_path": min_t or "",
                "descriptor_probe_path": desc_p or "",
                "descriptor_template_path": desc_t or ""
            })

            # progress print ogni 500 coppie
            if idx % 500 == 0 or idx == total:
                print(f"  processed {idx}/{total} pairs")

        except Exception as e:
            print(f"[ERRORE] pair {probe_base} vs {template_base}: {e}")
            traceback.print_exc()

    # salvo CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✅ Matching completato. Risultati salvati in: {output_csv}")
    return output_csv

if __name__ == "__main__":
    # esecuzione diretta per debug
    main()
