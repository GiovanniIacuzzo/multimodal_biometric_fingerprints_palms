import numpy as np
from clustering import consensus_kmeans

def assign_global_label(std_angle):
    if std_angle < 20: 
        return "Arch"
    elif std_angle < 45: 
        return "Loop"
    else: 
        return "Whorl"
    
def internal_clustering(features_dict, max_k_clusters):
    from numpy import array, bincount, where
    from pathlib import Path
    final_results = []
    for cls, img_feats_list in features_dict.items():
        feats_matrix = [f[1] for f in img_feats_list]
        img_paths = [f[0] for f in img_feats_list]

        cluster_labels, score = consensus_kmeans(feats_matrix, max_k_clusters)
        print(f"Class {cls} - best internal clusters: {len(set(cluster_labels))}, silhouette: {score:.2f}")

        # opzionale: unisci piccoli cluster
        counts = bincount(cluster_labels)
        small_clusters = where(counts < 3)[0]
        for sc in small_clusters:
            idxs = where(cluster_labels == sc)[0]
            for idx in idxs:
                distances = [np.linalg.norm(array(feats_matrix[idx]) - array(feats_matrix)[array(cluster_labels)==c].mean(axis=0))
                             for c in range(len(set(cluster_labels))) if c != sc]
                cluster_labels[idx] = np.argmin(distances)

        for path, lbl in zip(img_paths, cluster_labels):
            final_results.append((Path(path).name, str(path), cls, lbl))
    return final_results
