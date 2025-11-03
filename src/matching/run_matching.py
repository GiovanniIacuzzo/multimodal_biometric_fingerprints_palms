"""
run_matching.py
-----------------------------------
Esegue matching 1:N su tutto il dataset usando:
 - matcher_minutiae.minutiae_similarity (geometrico)
 - matcher_descriptor.descriptor_similarity (vettoriale)
 - score_fusion.adaptive_fusion (fusione adattiva)
"""

import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# tenta import config
try:
    from scripts.config import DATASET_DIR, FEATURES_DIR, RESULTS_DIR, ensure_dirs
except Exception:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET_DIR = os.path.join(BASE_DIR, "data")
    FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
    RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
    def ensure_dirs():
        os.makedirs(RESULTS_DIR, exist_ok=True)

from src.matching.matcher_minutiae import load_minutiae, minutiae_similarity
from src.matching.matcher_descriptor import descriptor_similarity
from src.matching.score_fusion import adaptive_fusion

MINUTIAE_DIR = os.path.join(FEATURES_DIR, "minutiae")
DESC_HAND_DIR = os.path.join(FEATURES_DIR, "descriptors_handcrafted")
DESC_DEEP_DIR = os.path.join(FEATURES_DIR, "descriptors_deep")
OUTPUT_SCORES_CSV = os.path.join(RESULTS_DIR, "scores.csv")


# ============================================================
#  Utility
# ============================================================
def find_file(base, dirs, patterns):
    for d in dirs:
        for p in patterns:
            path = os.path.join(d, p.format(base=base))
            if os.path.exists(path):
                return path
        gl = glob.glob(os.path.join(d, f"{base}*"))
        if gl:
            return gl[0]
    return None


def safe_load_minutiae(path):
    try:
        return load_minutiae(path)
    except Exception:
        return []


def compute_pair_scores(probe_id, template_id, subject_map):
    """Matching completo tra una coppia (probe, template)."""
    try:
        min_p = find_file(probe_id, [MINUTIAE_DIR], ["{base}_minutiae.json"])
        min_t = find_file(template_id, [MINUTIAE_DIR], ["{base}_minutiae.json"])
        desc_p = find_file(probe_id, [DESC_DEEP_DIR, DESC_HAND_DIR], ["{base}_features.npy", "{base}.npy"])
        desc_t = find_file(template_id, [DESC_DEEP_DIR, DESC_HAND_DIR], ["{base}_features.npy", "{base}.npy"])

        # Calcolo punteggi
        score_min = 0.0
        if min_p and min_t:
            probe_min = safe_load_minutiae(min_p)
            templ_min = safe_load_minutiae(min_t)
            if probe_min and templ_min:
                score_min = minutiae_similarity(templ_min, probe_min)

        score_desc = 0.0
        if desc_p and desc_t:
            score_desc = descriptor_similarity(desc_t, desc_p)

        score_final = adaptive_fusion(score_min, score_desc)

        match = 1 if subject_map.get(probe_id) == subject_map.get(template_id) else 0
        return {
            "probe_id": probe_id,
            "template_id": template_id,
            "score_minutiae": score_min,
            "score_descriptor": score_desc,
            "score_final": score_final,
            "match": match
        }
    except Exception as e:
        return {"probe_id": probe_id, "template_id": template_id, "error": str(e)}


# ============================================================
#  MAIN
# ============================================================
def main(catalog_path=None, output_csv=None, max_workers=8, limit_pairs=None):
    ensure_dirs()
    catalog_path = catalog_path or os.path.join(DATASET_DIR, "catalog.csv")
    output_csv = output_csv or OUTPUT_SCORES_CSV

    df = pd.read_csv(catalog_path)
    ids = df["image_id"].tolist()
    subject_map = dict(zip(df["image_id"], df["subject_id"]))
    pairs = [(p, t) for p in ids for t in ids]
    if limit_pairs:
        pairs = pairs[:limit_pairs]

    print(f"🔍 Matching: {len(ids)} immagini → {len(pairs)} coppie")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(compute_pair_scores, p, t, subject_map) for p, t in pairs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Matching in corso"):
            results.append(f.result())

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Matching completato. File salvato in: {output_csv}")


if __name__ == "__main__":
    main()
