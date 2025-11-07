import os
import cv2
import argparse
import time
import traceback
import numpy as np
from tqdm import tqdm
from config import config
from src.db.database import get_connection
from preprocessing.fingerprint_preprocess import preprocess_fingerprint


# ================================================
# UTILITY FUNZIONI
# ================================================

def load_image(path: str) -> np.ndarray:
    """Carica immagine in scala di grigi, controllando errori."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("Immagine non leggibile")
        return img
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere {path}: {e}")
        return None

def save_debug_images(results: dict, output_dir: str, base_name: str):
    """Salva tutte le fasi intermedie del preprocessing in formato sicuro."""
    os.makedirs(output_dir, exist_ok=True)

    def safe_convert(img: np.ndarray) -> np.ndarray:
        """Converte automaticamente in uint8 evitando warning."""
        if img.dtype == np.uint8:
            return img
        if img.dtype == bool:
            return (img.astype(np.uint8)) * 255
        if np.issubdtype(img.dtype, np.floating):
            # Se float tra 0-1 → scala a 0-255
            if img.max() <= 1.0:
                img = img * 255.0
            return np.clip(img, 0, 255).astype(np.uint8)
        if np.issubdtype(img.dtype, np.integer):
            return np.clip(img, 0, 255).astype(np.uint8)
        return img.astype(np.uint8)

    for key, img in results.items():
        if img is None:
            continue
        filename = os.path.join(output_dir, f"{base_name}_{key}.png")
        try:
            cv2.imwrite(filename, safe_convert(img))
        except Exception as e:
            print(f"[ATTENZIONE] Fallito salvataggio di {filename}: {e}")

def save_image_record(subject_id, filename, path_original, path_enhanced, orientation_mean=None, preprocessing_time=None):
    """Registra i metadati nel database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO images (subject_id, filename, path_original, path_enhanced, orientation_mean, preprocessing_time_sec, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (subject_id, filename, path_original, path_enhanced, orientation_mean, preprocessing_time, "done"))
    image_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return image_id

# ================================================
# MAIN PIPELINE
# ================================================
def run_preprocessing(
    input_dir: str = config.DATASET_DIR,
    output_dir: str = config.PROCESSED_DIR,
    debug: bool = True,
    small_subset: bool = False
):
    """
    Applica il preprocessing completo su un dataset di impronte digitali.
    Se small_subset=True, usa solo le prime 10 immagini per test/debug rapido.
    """
    # Filtra immagini valide
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    if not image_files:
        raise RuntimeError(f"Nessuna immagine trovata in {input_dir}")

    # Se richiesto, usa solo un piccolo subset
    if small_subset:
        image_files = image_files[:10]
        print(f"[DEBUG MODE] Utilizzo di solo {len(image_files)} immagini per test rapido.")

    print(f"Trovate {len(image_files)} immagini in '{input_dir}'")
    print(f"Output directory: {output_dir}")

    # Prepara le sottocartelle
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    if debug:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    # Loop principale di preprocessing
    for file_name in tqdm(image_files, desc="Preprocessing immagini", ncols=100):
        img_path = os.path.join(input_dir, file_name)
        img = load_image(img_path)
        if img is None:
            continue

        try:
            # Esegue preprocessing completo (enhancement, binarizzazione, thinning)
            start = time.time()
            results = preprocess_fingerprint(img, debug_dir=None)
            enhanced = results.get("enhanced", img)
            duration = time.time() - start

            # Salva immagine principale migliorata
            base_name = os.path.splitext(file_name)[0]
            out_path = os.path.join(output_dir, "enhanced", f"{base_name}_enhanced.png")
            cv2.imwrite(out_path, enhanced)

            # Se debug, salva le fasi intermedie
            if debug:
                debug_dir = os.path.join(output_dir, "debug", base_name)
                save_debug_images(results, debug_dir, base_name)

            # Salva nel DB (facoltativo)
            image_id = save_image_record(
                subject_id=None,  # opzionale: derivabile dal nome file
                filename=file_name,
                path_original=img_path,
                path_enhanced=out_path,
                orientation_mean=float(np.mean(results["orientation_map"])) if "orientation_map" in results else None,
                preprocessing_time=duration
            )
        except Exception as e:
            print(f"[ERRORE] Preprocessing fallito per {file_name}: {e}")
            traceback.print_exc()

    print("\nPreprocessing completato con successo!")
    print(f"Risultati salvati in: {output_dir}")


# ================================================
# ENTRY POINT
# ================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione pipeline di preprocessing impronte")
    parser.add_argument("--input", type=str, default=config.DATASET_DIR, help="Directory contenente le immagini grezze")
    parser.add_argument("--output", type=str, default=config.PROCESSED_DIR, help="Directory di output")
    parser.add_argument("--no-debug", action="store_true", help="Non salvare le immagini intermedie di debug")
    parser.add_argument("--small", action="store_true", help="Usa solo un piccolo subset di immagini per test")
    args = parser.parse_args()

    run_preprocessing(
        input_dir=args.input,
        output_dir=args.output,
        debug=not args.no_debug,
        small_subset=args.small
    )
