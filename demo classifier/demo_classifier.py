import cv2
import csv
from feature_fun import *
from classification import *
from visualization import *
from config import *

all_imgs = sorted(DATASET_DIR.glob("**/*.jpg"))
print(f"Found {len(all_imgs)} images.")

features_dict = {}
discarded = 0

for img_path in all_imgs:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        discarded += 1
        continue

    img_norm = normalize(img)
    std_ang = classify_std_angle(compute_orientation_field(img_norm))
    if std_ang < CONFIDENCE_THRESHOLD:
        discarded += 1
        continue

    global_label = assign_global_label(std_ang)
    feats = extract_multi_scale_features(img_norm)
    features_dict.setdefault(global_label, []).append((img_path, feats))

print(f"Discarded {discarded} images due to noise.")

final_results = internal_clustering(features_dict, max_k_clusters=MAX_K_CLUSTERS)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","path","global_class","cluster_in_class"])
    writer.writerows(final_results)
print(f"Saved stable complete labels to {OUTPUT_CSV}")

plot_class_distribution(final_results)
interactive_tsne(features_dict, final_results)
