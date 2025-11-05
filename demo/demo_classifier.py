import os
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === Importa moduli locali ===
from demo.feature_fun import (
    normalize,
    compute_orientation_field,
    classify_std_angle,
    extract_multi_scale_features,
    standardize_global_features,
)
from demo.clustering_utils import (
    assign_global_label,
    estimate_adaptive_thresholds,
    internal_clustering,
)
from demo.visualization import plot_class_distribution, interactive_tsne
from demo.config import DATASET_DIR, OUTPUT_CSV, MAX_K_CLUSTERS, CONFIDENCE_THRESHOLD

# ============================================================
# STEP 1 — CARICAMENTO IMMAGINI
# ============================================================
def load_and_preprocess_images(dataset_dir: Path):
    """
    Carica e normalizza tutte le immagini del dataset.
    Rimuove quelle troppo rumorose (soglia CONFIDENCE_THRESHOLD).
    """
    all_imgs = sorted(dataset_dir.glob("**/*.jpg"))
    print(f"🔍 Found {len(all_imgs)} images in {dataset_dir}")

    valid_images = []
    std_angles = []
    discarded = 0

    for img_path in tqdm(all_imgs, desc="Preprocessing images"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            discarded += 1
            continue

        img_norm = normalize(img)
        std_ang = classify_std_angle(compute_orientation_field(img_norm))

        # Scarta immagini troppo piatte o rumorose
        if std_ang < CONFIDENCE_THRESHOLD:
            discarded += 1
            continue

        valid_images.append((img_path, img_norm, std_ang))
        std_angles.append(std_ang)

    print(f"✅ Kept {len(valid_images)} valid images | ❌ Discarded {discarded} noisy images\n")
    return valid_images, np.array(std_angles)


# ============================================================
# STEP 2 — ESTRAZIONE FEATURE
# ============================================================
def extract_features(valid_images, adaptive_thresholds):
    """
    Estrae feature multi-scala e assegna la classe globale (Arch, Loop, Whorl)
    in base alle soglie adattive.
    """
    features_dict = {}

    for img_path, img_norm, std_ang in tqdm(valid_images, desc="Extracting features"):
        global_label = assign_global_label(std_ang, adaptive_thresholds)
        feats = extract_multi_scale_features(img_norm)
        features_dict.setdefault(global_label, []).append((img_path, feats))

    return features_dict


# ============================================================
# STEP 3 — PIPELINE COMPLETA
# ============================================================
def main():
    # === 1. Caricamento e pre-elaborazione ===
    valid_images, std_angles = load_and_preprocess_images(DATASET_DIR)

    # === 2. Calcolo soglie adattive ===
    adaptive_thresholds = estimate_adaptive_thresholds(std_angles)
    print(f"📈 Adaptive thresholds: t1={adaptive_thresholds[0]:.2f}, t2={adaptive_thresholds[1]:.2f}\n")

    # === 3. Estrazione feature multi-scala e classificazione globale ===
    features_dict = extract_features(valid_images, adaptive_thresholds)

    # === 4. Standardizzazione globale ===
    features_dict, _ = standardize_global_features(features_dict)
    print("⚙️  Global standardization completed.\n")

    # === 5. Clustering interno per ogni classe ===
    final_results = internal_clustering(features_dict, max_k_clusters=MAX_K_CLUSTERS)
    print("✅ Internal clustering completed.\n")

    # === 6. Salvataggio CSV finale ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "global_class", "cluster_in_class"])
        writer.writerows(final_results)
    print(f"💾 Saved final labeled dataset to {OUTPUT_CSV}\n")

    # === 7. Visualizzazioni (facoltative) ===
    try:
        plot_class_distribution(final_results)
        interactive_tsne(features_dict, final_results)
    except Exception as e:
        print(f"⚠️ Visualization skipped due to: {e}")


if __name__ == "__main__":
    main()
