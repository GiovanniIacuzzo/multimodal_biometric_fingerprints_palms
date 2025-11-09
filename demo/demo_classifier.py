import os
import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from demo.feature_fun import (
    normalize,
    auto_rotate_90,
    compute_orientation_field,
    classify_std_angle,
    extract_rotation_invariant_features
    )
from demo.clustering_utils import (
    assign_global_label,
    estimate_adaptive_thresholds,
    internal_clustering,
    standardize_global_features
)
from demo.visualization import plot_class_distribution, interactive_tsne
from demo.config import DATASET_DIR, OUTPUT_CSV, MAX_K_CLUSTERS, CONFIDENCE_THRESHOLD

def load_and_preprocess_images(dataset_dir: Path):
    """
    Carica e normalizza tutte le immagini del dataset.
    Applica auto-rotazione in step di 90°. Raggruppa immagini per ID.
    """
    all_imgs = sorted(dataset_dir.glob("**/*.jpg"))
    print(f"Found {len(all_imgs)} images in {dataset_dir}")

    id_groups = {}
    discarded = 0
    kept = 0

    for img_path in tqdm(all_imgs, desc="Preprocessing images"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) if 'cv2' in globals() else None
        import cv2 as _cv2
        img = _cv2.imread(str(img_path), _cv2.IMREAD_GRAYSCALE)
        if img is None:
            discarded += 1
            continue

        img_norm = normalize(img)
        img_rot = auto_rotate_90(img_norm)
        std_ang = classify_std_angle(compute_orientation_field(img_rot))

        if std_ang < CONFIDENCE_THRESHOLD:
            discarded += 1
            continue

        fname = img_path.stem
        file_id = fname.split("_")[0].lstrip("0") or fname.split("_")[0]
        id_groups.setdefault(file_id, []).append((img_path, img_rot, std_ang))
        kept += 1

    print(f"Kept {kept} valid images | Discarded {discarded} noisy images\n")
    return id_groups

# ----------------------------
# STEP 2: Calcolo soglie adattive e assegnazione classe per ID
# ----------------------------
def assign_labels_by_id(id_groups):
    id_std = {fid: float(np.mean([s for _, _, s in imgs])) for fid, imgs in id_groups.items()}
    adaptive_thresholds = estimate_adaptive_thresholds(list(id_std.values()))
    print(f"Adaptive thresholds (ID-level): t1={adaptive_thresholds[0]:.2f}, t2={adaptive_thresholds[1]:.2f}\n")
    id_labels = {fid: assign_global_label(std, adaptive_thresholds) for fid, std in id_std.items()}
    return id_labels, adaptive_thresholds

# ----------------------------
# STEP 3: Estrazione feature per ID
# ----------------------------
def extract_features_by_id(id_groups, id_labels):
    """
    Restituisce features_dict: {class: [(id, feat), ...]}
    Le feature sono la media delle feature rotation-invariant delle immagini appartenenti allo stesso ID.
    """
    features_dict = {}
    for fid, imgs in tqdm(id_groups.items(), desc="Extracting ID features"):
        feats_all = []
        for img_path, img, std in imgs:
            feat = extract_rotation_invariant_features(img)
            feats_all.append(feat)
        if not feats_all:
            continue
        mean_feat = np.mean(np.stack(feats_all, axis=0), axis=0)
        cls = id_labels[fid]
        features_dict.setdefault(cls, []).append((fid, mean_feat.astype(np.float32)))
    return features_dict

# ----------------------------
# STEP 4: Pipeline principale
# ----------------------------
def main():
    os.makedirs(os.path.dirname(str(OUTPUT_CSV)), exist_ok=True)

    id_groups = load_and_preprocess_images(DATASET_DIR)
    if not id_groups:
        print("Nessuna immagine valida trovata. Esco.")
        return

    id_labels, adaptive_thresholds = assign_labels_by_id(id_groups)
    features_dict = extract_features_by_id(id_groups, id_labels)

    # Standardizzazione globale
    features_dict, scaler = standardize_global_features(features_dict)
    print("Global standardization completed.\n")

    # Clustering interno (a livello ID)
    final_id_results = internal_clustering(features_dict, max_k_clusters=MAX_K_CLUSTERS)
    print("Internal clustering completed.\n")

    rows = []
    # final_id_results: [(id, class, cluster)]
    id_to_cluster = {fid: (cls, cluster) for fid, cls, cluster in final_id_results}
    for fid, imgs in id_groups.items():
        if fid not in id_to_cluster:
            cls = id_labels.get(fid, "Unknown")
            cluster = 0
        else:
            cls, cluster = id_to_cluster[fid]
        for img_path, _, _ in imgs:
            rows.append([img_path.name, str(img_path), cls, int(cluster)])

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "global_class", "cluster_in_class"])
        writer.writerows(rows)
    print(f"Saved final labeled dataset to {OUTPUT_CSV}\n")

    # Visualizzazioni (facoltative)
    # plot_class_distribution(rows)
    # interactive_tsne(features_dict, final_id_results, FIGURE_DIR=OUTPUT_CSV)


if __name__ == "__main__":
    main()
