import os
import cv2
import argparse
import time
import traceback
import numpy as np
from tqdm import tqdm
from config import config_fingerprint
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.preprocessing.fingerprint_preprocess import preprocess_fingerprint
import logging
from src.db.database import save_image_record, ensure_subject

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ====================================================
# UTILITY FUNZIONI
# ====================================================

def load_image(path: str) -> np.ndarray:
    """Carica immagine in scala di grigi, con controllo errori."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("Immagine non leggibile")
        return img
    except Exception as e:
        logging.error(f"Impossibile leggere {path}: {e}")
        return None


def save_debug_images(results: dict, debug_base: str, base_name: str):
    """
    Salva le immagini di debug in:
    debug/<cluster>/<tipo>/<file>.jpg
    """
    for key, img in results.items():
        if img is None:
            continue

        out_dir = os.path.join(debug_base, key)
        os.makedirs(out_dir, exist_ok=True)

        filename = os.path.join(out_dir, f"{base_name}.jpg")

        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        elif np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

        cv2.imwrite(filename, img)


# ====================================================
# MAIN PIPELINE
# ====================================================

def run_preprocessing(
    input_dir: str = config_fingerprint.DATASET_DIR,
    output_dir: str = config_fingerprint.PROCESSED_DIR,
    debug: bool = True,
    small_subset: bool = False,
    max_workers: int = 4
):
    """Applica preprocessing completo a un dataset di impronte digitali."""

    # ====================================================
    # SCANSIONE IMMAGINI
    # ====================================================
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, f))

    if not image_files:
        raise RuntimeError(f"Nessuna immagine trovata in {input_dir}")

    if small_subset:
        image_files = image_files[:10]
        logging.info(f"[DEBUG MODE] Uso solo {len(image_files)} immagini.")

    logging.info(f"Trovate {len(image_files)} immagini nel dataset.")

    enhanced_dir = os.path.join(output_dir, "enhanced")
    debug_root = os.path.join(output_dir, "debug") if debug else None

    os.makedirs(enhanced_dir, exist_ok=True)
    if debug:
        os.makedirs(debug_root, exist_ok=True)

    # ====================================================
    # FUNZIONE PER PROCESSARE UNA SINGOLA IMMAGINE
    # ====================================================
    def process_single_image(img_path: str):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]

        img = load_image(img_path)
        if img is None:
            return None, file_name

        try:
            rel_dir = os.path.relpath(os.path.dirname(img_path), input_dir)

            enhanced_subdir = os.path.join(enhanced_dir, rel_dir)
            os.makedirs(enhanced_subdir, exist_ok=True)

            if debug:
                debug_base = os.path.join(debug_root, rel_dir)
                os.makedirs(debug_base, exist_ok=True)
            else:
                debug_base = None

            start = time.time()
            results = preprocess_fingerprint(img, debug_dir=None)
            elapsed = time.time() - start

            enhanced_img = results.get("enhanced", img)
            enhanced_path = os.path.join(enhanced_subdir, f"{base_name}_enhanced.jpg")
            cv2.imwrite(enhanced_path, enhanced_img)

            skeleton = results.get("skeleton")
            if skeleton is not None:
                skeleton_path = os.path.join(enhanced_subdir, f"{base_name}_skeleton.jpg")
                cv2.imwrite(skeleton_path, skeleton)
            else:
                skeleton_path = None

            if debug:
                save_debug_images(results, debug_base, base_name)

            subject_id = None
            if "_" in base_name:
                subject_code = base_name.split("_")[0]
                subject_id = ensure_subject(subject_code)

            image_id = save_image_record(
                subject_id=subject_id,
                filename=f"{base_name}.jpg",
                path_original=img_path,
                path_enhanced=enhanced_path,
                path_skeleton=skeleton_path,
                orientation_mean=float(np.mean(results["orientation_map"])) if "orientation_map" in results else None,
                preprocessing_time=elapsed
            )

            return image_id, file_name

        except Exception as e:
            logging.error(f"Errore nel preprocessing di {file_name}: {e}")
            traceback.print_exc()
            return None, file_name

    # ====================================================
    # EXECUTOR PARALLELO
    # ====================================================
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, f): f for f in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing", ncols=90):
            image_id, file_name = future.result()
            if image_id is None:
                logging.warning(f"Immagine NON processata: {file_name}")

    logging.info("Preprocessing completato!")
    logging.info(f"Risultati salvati in: {output_dir}")


# ====================================================
# ENTRY POINT
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=config_fingerprint.SORTED_DATASET_DIR)
    parser.add_argument("--output", type=str, default=config_fingerprint.PROCESSED_DIR)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    run_preprocessing(
        input_dir=args.input,
        output_dir=args.output,
        debug=not args.no_debug,
        small_subset=args.small
    )
