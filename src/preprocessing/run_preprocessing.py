import os
import cv2
import argparse
import time
import traceback
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore, Style
import logging

from config import config_fingerprint
from src.preprocessing.fingerprint_preprocess import preprocess_fingerprint

# ====================================================
# LOGGING SETUP
# ====================================================
log_dir = "data/metadata"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "preprocessing.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ====================================================
# UTILITY
# ====================================================
def console_step(title: str):
    """Stampa un titolo colorato per una sezione della pipeline."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

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
    """Salva le immagini di debug in debug/<tipo>/<file>.jpg"""
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
    debug: bool = False,
    small_subset: bool = False,
    max_workers: int = 4
):
    console_step("Inizio Preprocessing Impronte")

    # --- Scansione immagini ---
    VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.lower().endswith(VALID_EXTS)
    ]

    if not image_files:
        logging.error(f"Nessuna immagine trovata in {input_dir}")
        raise RuntimeError(f"Nessuna immagine trovata in {input_dir}")

    if small_subset:
        image_files = image_files[:10]
        logging.info(f"[DEBUG MODE] Uso solo {len(image_files)} immagini.")

    logging.info(f"Trovate {len(image_files)} immagini nel dataset.")
    print(f"→ {len(image_files)} immagini trovate in {input_dir}")

    enhanced_dir = os.path.join(output_dir, "enhanced")
    debug_root = os.path.join(output_dir, "debug") if debug else None

    os.makedirs(enhanced_dir, exist_ok=True)
    if debug:
        os.makedirs(debug_root, exist_ok=True)

    # --- Funzione per processare una singola immagine ---
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

            debug_base = os.path.join(debug_root, rel_dir) if debug else None
            if debug and debug_base:
                os.makedirs(debug_base, exist_ok=True)

            start = time.time()
            results = preprocess_fingerprint(img, debug_dir=None)
            elapsed = time.time() - start

            # Salvataggio immagini preprocessate
            enhanced_img = results.get("enhanced", img)
            enhanced_path = os.path.join(enhanced_subdir, f"{base_name}_enhanced.jpg")
            cv2.imwrite(enhanced_path, enhanced_img)

            skeleton = results.get("skeleton")
            skeleton_path = None
            if skeleton is not None:
                skeleton_path = os.path.join(enhanced_subdir, f"{base_name}_skeleton.jpg")
                cv2.imwrite(skeleton_path, skeleton)

            if debug and debug_base:
                save_debug_images(results, debug_base, base_name)

            logging.info(f"✔ Processata: {file_name} ({elapsed:.2f}s)")
            return image_id, file_name

        except Exception as e:
            logging.error(f"Errore nel preprocessing di {file_name}: {e}")
            traceback.print_exc()
            return None, file_name

    # --- Parallel processing ---
    console_step("Esecuzione Preprocessing in parallelo")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, f): f for f in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing", ncols=90):
            image_id, file_name = future.result()
            if image_id is None:
                logging.warning(f"Immagine NON processata: {file_name}")

    console_step("Preprocessing completato")
    logging.info(f"Risultati salvati in: {output_dir}")
    print(f"{Fore.GREEN}✨ Pipeline completata con successo! ✨{Style.RESET_ALL}")
    

# ====================================================
# ENTRYPOINT
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
        output_dir=args.output)
