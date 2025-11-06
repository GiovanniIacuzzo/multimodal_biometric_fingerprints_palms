import os
import cv2
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from src.preprocessing.enhancement import preprocess_fingerprint

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
    """Salva tutte le fasi intermedie del preprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    for key, img in results.items():
        if img is None:
            continue
        filename = os.path.join(output_dir, f"{base_name}_{key}.png")
        try:
            cv2.imwrite(filename, img)
        except Exception:
            print(f"[ATTENZIONE] Fallito salvataggio di {filename}")



# ================================================
# MAIN PIPELINE
# ================================================

def run_preprocessing(
    input_dir: str,
    output_dir: str = "outputs/preprocessed",
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
            results = preprocess_fingerprint(img, debug_dir=None)
            enhanced = results.get("enhanced", img)

            # Salva immagine principale migliorata
            base_name = os.path.splitext(file_name)[0]
            out_path = os.path.join(output_dir, "enhanced", f"{base_name}_enhanced.png")
            cv2.imwrite(out_path, enhanced)

            # Se debug, salva le fasi intermedie
            if debug:
                debug_dir = os.path.join(output_dir, "debug", base_name)
                save_debug_images(results, debug_dir, base_name)

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
    parser.add_argument("--input", type=str, required=True, help="Directory contenente le immagini grezze")
    parser.add_argument("--output", type=str, default="outputs/preprocessed", help="Directory di output")
    parser.add_argument("--no-debug", action="store_true", help="Non salvare le immagini intermedie di debug")
    args = parser.parse_args()

    run_preprocessing(
        input_dir=args.input,
        output_dir=args.output,
        debug=not args.no_debug
    )
