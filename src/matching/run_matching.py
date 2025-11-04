import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# Tenta import config
try:
    from config.config import DATASET_DIR, FEATURES_DIR, RESULTS_DIR, ensure_dirs
except Exception:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET_DIR = os.path.join(BASE_DIR, "data")
    FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
    RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
    def ensure_dirs():
        os.makedirs(RESULTS_DIR, exist_ok=True)

from src.matching.matcher_minutiae import load_minutiae, minutiae_similarity
from src.matching.matcher_descriptor import descriptor_similarity, safe_load_descriptor
from src.matching.score_fusion import adaptive_fusion

MINUTIAE_DIR = os.path.join(FEATURES_DIR, "minutiae")
DESC_DIR = MINUTIAE_DIR  # Se i descriptor sono nello stesso posto
OUTPUT_SCORES_CSV = os.path.join(RESULTS_DIR, "scores.csv")

# ============================================================
# Utility
# ============================================================
def find_file(base, dirs, patterns):
    """Cerca un file in più directory e pattern; fallback con glob."""
    for d in dirs:
        for p in patterns:
            path = os.path.join(d, p.format(base=base))
            if os.path.exists(path):
                return path
        gl = glob.glob(os.path.join(d, f"{base}*"))
        if gl:
            return gl[0]
    print(f"⚠️ Attenzione: file per '{base}' non trovato in {dirs}")
    return None

def safe_load_minutiae(path):
    try:
        minutiae = load_minutiae(path)
        if not minutiae:
            print(f"⚠️ File vuoto: {path}")
        return minutiae
    except Exception as e:
        print(f"❌ Errore caricando minutiae da {path}: {e}")
        return []

# ============================================================
# Matching
# ============================================================
def compute_pair_scores(probe_id, template_id, subject_map):
    """Matching completo tra una coppia (probe, template)."""
    try:
        # === Percorsi file ===
        min_p = find_file(probe_id, [MINUTIAE_DIR], ["{base}_minutiae.json"])
        min_t = find_file(template_id, [MINUTIAE_DIR], ["{base}_minutiae.json"])
        desc_p = find_file(probe_id, [DESC_DIR], ["{base}_descriptor.npy"])
        desc_t = find_file(template_id, [DESC_DIR], ["{base}_descriptor.npy"])

        # === Punteggio minutiae ===
        score_min = 0.0
        if min_p and min_t:
            probe_min = safe_load_minutiae(min_p)
            templ_min = safe_load_minutiae(min_t)
            print(f"DEBUG: probe {probe_id}={len(probe_min)} minutiae, template {template_id}={len(templ_min)} minutiae")
            if probe_min and templ_min:
                score_min = minutiae_similarity(templ_min, probe_min)

        # === Punteggio descriptor ===
        score_desc = 0.0
        if desc_p and desc_t:
            desc_arr_p = safe_load_descriptor(desc_p)
            desc_arr_t = safe_load_descriptor(desc_t)
            if desc_arr_p.size > 0 and desc_arr_t.size > 0:
                score_desc = descriptor_similarity(desc_arr_t, desc_arr_p)

        # === Fusione punteggi ===
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
        print(f"❌ Errore matching {probe_id} vs {template_id}: {e}")
        return {
            "probe_id": probe_id,
            "template_id": template_id,
            "score_minutiae": 0.0,
            "score_descriptor": 0.0,
            "score_final": 0.0,
            "match": 0,
            "error": str(e)
        }

# ============================================================
# MAIN
# ============================================================
def main(catalog_path=None, output_csv=None, max_workers=8, limit_pairs=None, test_mode=False):
    ensure_dirs()
    catalog_path = catalog_path or os.path.join(DATASET_DIR, "catalog.csv")
    output_csv = output_csv or OUTPUT_SCORES_CSV

    df = pd.read_csv(catalog_path)

    if test_mode:
        df = df.head(10)
        print(f"⚙️ Modalità DEMO attiva: verranno processate solo {len(df)} immagini.")

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

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue matching 1:N su fingerprint.")
    parser.add_argument("--test", action="store_true", help="Modalità demo: usa solo 10 immagini")
    args = parser.parse_args()
    main(test_mode=args.test)
