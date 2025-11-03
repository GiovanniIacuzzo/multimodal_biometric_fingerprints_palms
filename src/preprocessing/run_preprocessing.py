"""
run_preprocessing.py (robust runner)
-----------------------------------
Esegue preprocessing su immagini elencate in catalog.csv.
Cose nuove:
- argparse per path e limit
- salvataggio incrementale del log (checkpoint)
- preview composito per debug (normalized | enhanced | binary | skeleton)
- metriche corrette e calcolate sulle immagini mascherate dalla ROI
"""

import os
import cv2
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.preprocessing.enhancement import preprocess_fingerprint

# -------------------------
# Utils & CONFIG DEFAULTS
# -------------------------
DEFAULT_CATALOG = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/metadata/catalog.csv"
DEFAULT_OUTPUT = "/Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/data/processed"
DEFAULT_LOG = "preprocessing_log.csv"
CHECKPOINT_EVERY = 50  # salva log ogni N immagini


def is_valid(img: np.ndarray, min_std: float = 8.0) -> bool:
    """Quick sanity: non vuota e non completamente saturata; std minimo."""
    if img is None:
        return False
    if img.size == 0:
        return False
    mean, std = float(np.mean(img)), float(np.std(img))
    if std < min_std:  # immagine troppo piatta
        return False
    if mean < 5 or mean > 250:
        # immagine completamente nera/bianca
        return False
    return True


def enhance_input_image(img: np.ndarray) -> np.ndarray:
    """Pre-enhancement: CLAHE + light denoising."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    den = cv2.fastNlMeansDenoising(img_clahe, h=8, templateWindowSize=7, searchWindowSize=21)
    return den


def compute_quality_metrics(results: dict) -> dict:
    """
    Calcola metriche utili:
      - contrast = std(normalized)
      - ridge_valley_ratio = mean(enhanced where ridge) / mean(enhanced where non-ridge within ROI)
      - skeleton_density = (#skeleton pixels) / (#ROI pixels)
      - snr_estimate = mean_ridge / (std_valley + eps)
    """
    enhanced = results.get("enhanced_masked", results.get("enhanced"))
    binary = results["binary"]
    skeleton = results["skeleton"]
    roi = results["roi_mask"]

    # ensure arrays proper dtype
    enhanced = (enhanced.astype(np.float32) if enhanced is not None else None)
    binary_bool = (binary > 0)

    # Contrast metric (use normalized image if available)
    normalized = results.get("normalized")
    contrast = float(np.std(normalized)) if normalized is not None else float(np.std(enhanced))

    # compute mean intensities
    eps = 1e-6
    if np.any(binary_bool):
        ridge_mean = float(np.mean(enhanced[binary_bool]))
        # valley: pixels in ROI but not ridge
        if roi is not None and roi.sum() > 0:
            roi_bool = (roi > 0)
            valley_mask = roi_bool & (~binary_bool)
            if np.any(valley_mask):
                valley_mean = float(np.mean(enhanced[valley_mask]))
            else:
                # fallback to whole image non-ridge
                valley_mean = float(np.mean(enhanced[~binary_bool]))
        else:
            valley_mean = float(np.mean(enhanced[~binary_bool]))
    else:
        # no ridge found: set defaults
        ridge_mean = 0.0
        valley_mean = float(np.mean(enhanced)) if enhanced is not None else 1.0

    ridge_valley_ratio = (ridge_mean + eps) / (valley_mean + eps)
    # skeleton density w.r.t ROI area (avoid dividing by full image)
    roi_area = (roi > 0).sum() if roi is not None else binary.size
    skeleton_pixels = (skeleton > 0).sum()
    skeleton_density = float(skeleton_pixels) / float(max(roi_area, 1))

    snr_estimate = (ridge_mean) / (np.std(enhanced[~binary_bool]) + eps) if np.any(~binary_bool) else 0.0

    return {
        "contrast": round(contrast, 3),
        "ridge_mean": round(ridge_mean, 3),
        "valley_mean": round(valley_mean, 3),
        "ridge_valley_ratio": round(ridge_valley_ratio, 3),
        "skeleton_density": round(skeleton_density, 5),
        "snr_estimate": round(float(snr_estimate), 3)
    }


def make_preview(results: dict, max_w=800) -> np.ndarray:
    """Crea un'immagine composita per debug: normalized | enhanced | binary | skeleton"""
    parts = []
    for key in ["normalized", "enhanced", "binary", "skeleton"]:
        img = results.get(key)
        if img is None:
            parts.append(np.zeros((64, 64), dtype=np.uint8))
        else:
            # ensure grayscale uint8
            if img.dtype != np.uint8:
                tmp = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                tmp = img
            parts.append(tmp)
    # resize rows to same height
    h = min(p.shape[0] for p in parts)
    resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in parts]
    composite = np.hstack(resized)
    # scale to max_w if needed
    if composite.shape[1] > max_w:
        scale = max_w / composite.shape[1]
        composite = cv2.resize(composite, (max_w, int(composite.shape[0] * scale)))
    return composite


# -------------------------
# Main flow
# -------------------------
def process_catalog(catalog_path: str, output_dir: str, log_path: str, limit: int = None):
    os.makedirs(output_dir, exist_ok=True)
    log_full_path = os.path.join(output_dir, log_path)

    df = pd.read_csv(catalog_path)
    if limit is not None:
        df = df.iloc[:limit]

    rows = []
    # If checkpoint exists, load to continue
    if os.path.exists(log_full_path):
        done = pd.read_csv(log_full_path)
        processed_set = set(done['image_id'].astype(str).tolist())
    else:
        processed_set = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        image_path = row['path']
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        if image_id in processed_set:
            continue

        save_dir = os.path.join(output_dir, image_id)
        os.makedirs(save_dir, exist_ok=True)

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if not is_valid(img):
                rows.append({
                    "image_id": image_id,
                    "status": "INVALID_IMAGE"
                })
                # save small placeholder
                continue

            pre = enhance_input_image(img)
            results = preprocess_fingerprint(pre)

            # Ensure all arrays uint8 and save them
            for key, arr in results.items():
                if arr is None:
                    continue
                out = arr
                if out.dtype != np.uint8:
                    out = cv2.normalize(out.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{key}.png"), out)

            # save preview composite aggiornato
            preview_parts = []
            for key in ["normalized", "smoothed", "enhanced", "binary", "skeleton"]:
                img_part = results.get(key)
                if img_part is None:
                    img_part = np.zeros((64, 64), dtype=np.uint8)
                if img_part.dtype != np.uint8:
                    img_part = cv2.normalize(img_part.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                preview_parts.append(img_part)

            # resize to same height
            h = min(p.shape[0] for p in preview_parts)
            resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in preview_parts]
            composite = np.hstack(resized)

            # scale preview if too wide
            max_w = 1200
            if composite.shape[1] > max_w:
                scale = max_w / composite.shape[1]
                composite = cv2.resize(composite, (max_w, int(composite.shape[0] * scale)))

            cv2.imwrite(os.path.join(save_dir, "preview.png"), composite)

            metrics = compute_quality_metrics(results)

            row_log = {
                "image_id": image_id,
                "status": "OK",
                **metrics
            }
            rows.append(row_log)

        except Exception as ex:
            rows.append({
                "image_id": image_id,
                "status": f"ERROR: {str(ex)}"
            })

        # checkpoint save every N images
        if len(rows) % CHECKPOINT_EVERY == 0:
            df_log = pd.DataFrame(rows)
            if os.path.exists(log_full_path):
                existing = pd.read_csv(log_full_path)
                df_out = pd.concat([existing, df_log], ignore_index=True)
            else:
                df_out = df_log
            df_out.to_csv(log_full_path, index=False)

    # final write
    df_log = pd.DataFrame(rows)
    if os.path.exists(log_full_path):
        existing = pd.read_csv(log_full_path)
        df_out = pd.concat([existing, df_log], ignore_index=True)
    else:
        df_out = df_log
    df_out.to_csv(log_full_path, index=False)
    print(f"Logs salvati in: {log_full_path}")

def main(catalog_path: str = DEFAULT_CATALOG,
         output_dir: str = DEFAULT_OUTPUT,
         log_path: str = DEFAULT_LOG,
         limit: int = None):
    """
    Esegue l'intero processo di preprocessing:
    - Legge il catalogo
    - Applica la pipeline a ciascuna immagine
    - Salva risultati e log
    """
    print("\n🧼 Avvio preprocessing immagini...")
    process_catalog(catalog_path, output_dir, log_path, limit)
    print("✅ Preprocessing completato con successo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint preprocessing runner")
    parser.add_argument("--catalog", type=str, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=str, default=DEFAULT_LOG)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(
        catalog_path=args.catalog,
        output_dir=args.out,
        log_path=args.log,
        limit=args.limit
    )

