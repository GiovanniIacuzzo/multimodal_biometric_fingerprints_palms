import os
import cv2
import argparse
import time
import traceback
import numpy as np
from tqdm import tqdm
from config import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.preprocessing.fingerprint_preprocess import preprocess_fingerprint
import logging
from src.db.database import save_image_record, ensure_subject

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ====================================================
# UTILITY FUNZIONI
# ====================================================

def load_image(path: str) -> np.ndarray:
    """Carica immagine in scala di grigi, controllando errori."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("Immagine non leggibile")
        return img
    except Exception as e:
        logging.error(f"Impossibile leggere {path}: {e}")
        return None

def save_debug_images(results: dict, output_dir: str, base_name: str):
    """Salva tutte le fasi intermedie del preprocessing in formato sicuro."""
    os.makedirs(output_dir, exist_ok=True)

    def safe_convert(img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if img.dtype == bool:
            return (img.astype(np.uint8)) * 255
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = img * 255.0
            return np.clip(img, 0, 255).astype(np.uint8)
        if np.issubdtype(img.dtype, np.integer):
            return np.clip(img, 0, 255).astype(np.uint8)
        return img.astype(np.uint8)

    for key, img in results.items():
        if img is None:
            continue
        filename = os.path.join(output_dir, f"{base_name}_{key}.jpg")
        try:
            cv2.imwrite(filename, safe_convert(img))
        except Exception as e:
            logging.warning(f"Fallito salvataggio di {filename}: {e}")

# ====================================================
# MAIN PIPELINE
# ====================================================
def run_preprocessing(
    input_dir: str = config.DATASET_DIR,
    output_dir: str = config.PROCESSED_DIR,
    debug: bool = True,
    small_subset: bool = False,
    max_workers: int = 4  # numero di thread paralleli
):
    """Applica il preprocessing completo su un dataset di impronte digitali."""
    # Filtra immagini valide
    image_files = sorted([f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if not image_files:
        raise RuntimeError(f"Nessuna immagine trovata in {input_dir}")

    if small_subset:
        image_files = image_files[:10]
        logging.info(f"[DEBUG MODE] Uso solo {len(image_files)} immagini per test rapido.")
        logging.info("Le immagini sono: " + ", ".join(image_files))

    logging.info(f"Trovate {len(image_files)} immagini in '{input_dir}'")
    logging.info(f"Output directory: {output_dir}")

    # Prepara sottocartelle
    enhanced_dir = os.path.join(output_dir, "enhanced")
    debug_dir_base = os.path.join(output_dir, "debug") if debug else None
    os.makedirs(enhanced_dir, exist_ok=True)
    if debug:
        os.makedirs(debug_dir_base, exist_ok=True)

    def process_single_image(file_name: str):
        img_path = os.path.join(input_dir, file_name)
        img = load_image(img_path)
        if img is None:
            return None, file_name

        try:
            start_time = time.time()
            results = preprocess_fingerprint(img, debug_dir=None)
            enhanced = results.get("enhanced", img)
            duration = time.time() - start_time

            base_name = os.path.splitext(file_name)[0]
            out_path = os.path.join(enhanced_dir, f"{base_name}_enhanced.jpg")
            cv2.imwrite(out_path, enhanced)

            if debug:
                save_debug_images(results, os.path.join(debug_dir_base, base_name), base_name)

            # Recupero o creazione subject_id
            subject_id = None
            if "_" in base_name:
                subject_code = base_name.split("_")[0]
                subject_id = ensure_subject(subject_code)

            # Salvataggio sicuro nel DB
            image_id = save_image_record(
                subject_id=subject_id,
                filename=f"{base_name}.jpg",
                path_original=img_path,
                path_enhanced=out_path,
                path_skeleton=os.path.join(debug_dir_base, base_name),
                orientation_mean=float(np.mean(results["orientation_map"])) if "orientation_map" in results else None,
                preprocessing_time=duration
            )
            return image_id, file_name

        except Exception as e:
            logging.error(f"Preprocessing fallito per {file_name}: {e}")
            traceback.print_exc()
            return None, file_name

    # =======================================
    # Esecuzione parallela
    # =======================================
    results_summary = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, f): f for f in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing immagini", ncols=100):
            image_id, file_name = future.result()
            if image_id is None:
                logging.warning(f"Immagine '{file_name}' NON processata!")

    logging.info("Preprocessing completato con successo!")
    logging.info(f"Risultati salvati in: {output_dir}")

# ====================================================
# ENTRY POINT
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline di preprocessing impronte")
    parser.add_argument("--input", type=str, default=config.DATASET_DIR, help="Directory immagini grezze")
    parser.add_argument("--output", type=str, default=config.PROCESSED_DIR, help="Directory output")
    parser.add_argument("--no-debug", action="store_true", help="Non salvare immagini di debug")
    parser.add_argument("--small", action="store_true", help="Usa subset di immagini per test rapido")
    args = parser.parse_args()

    run_preprocessing(
        input_dir=args.input,
        output_dir=args.output,
        debug=not args.no_debug,
        small_subset=args.small
    )
