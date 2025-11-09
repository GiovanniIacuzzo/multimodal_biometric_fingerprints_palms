import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path
from skimage.feature import local_binary_pattern

# =========================
# FEATURE EXTRA: HOG + Gabor + LBP
# =========================
def extract_multi_scale_features(img):
    import cv2
    from skimage.feature import hog
    feats = []

    scales = [1.0, 0.75, 0.5]
    for s in scales:
        resized = cv2.resize(img, None, fx=s, fy=s)
        # HOG
        h = hog(resized, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, feature_vector=True)
        feats.append(h.astype(np.float32))
        # LBP
        lbp = local_binary_pattern(resized, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
        feats.append(hist.astype(np.float32))
    feats = np.concatenate(feats, axis=0)
    return feats

# =========================
# GLOBAL STANDARDIZATION
# =========================
def standardize_global_features(features_dict):
    all_feats = []
    for cls, lst in features_dict.items():
        for _, f in lst:
            all_feats.append(f)
    if not all_feats:
        return features_dict, None

    max_len = max(f.shape[0] for f in all_feats)
    stacked = np.zeros((len(all_feats), max_len), dtype=np.float32)
    for i, f in enumerate(all_feats):
        stacked[i, :f.shape[0]] = f

    if stacked.shape[0] < 2:
        return features_dict, None

    scaler = StandardScaler().fit(stacked)
    for cls, img_feats_list in features_dict.items():
        for i, (path, feat) in enumerate(img_feats_list):
            padded = np.zeros((max_len,), dtype=np.float32)
            padded[:feat.shape[0]] = feat
            scaled = scaler.transform(padded.reshape(1,-1)).flatten()
            features_dict[cls][i] = (path, scaled[:feat.shape[0]])
    return features_dict, scaler

# =========================
# CONSENSUS KMEANS
# =========================
def consensus_kmeans(feats_matrix, max_k=5, n_repeats=5, min_cluster_size=3):
    n_samples = feats_matrix.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int), 0.0

    feats_std = StandardScaler().fit_transform(feats_matrix)
    feats_pca = PCA(n_components=min(30, feats_std.shape[1], n_samples)).fit_transform(feats_std)

    best_score = -1.0
    best_labels = np.zeros(n_samples, dtype=int)
    max_k_test = min(max_k, n_samples-1)

    for k in range(2, max_k_test+1):
        all_labels = []
        for seed in range(n_repeats):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            lbl = km.fit_predict(feats_pca)
            all_labels.append(lbl)
        all_labels = np.array(all_labels)
        labels = np.array([np.bincount(all_labels[:,i]).argmax() for i in range(n_samples)])
        try:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(feats_pca, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    # fallback DBSCAN
    if best_score < 0.25:
        db = DBSCAN(eps=0.8, min_samples=3).fit(feats_pca)
        db_labels = db.labels_
        if len(set(db_labels)) > 1:
            unique = [c for c in set(db_labels) if c != -1]
            centroids = {c: feats_pca[db_labels==c].mean(axis=0) for c in unique}
            for i, l in enumerate(db_labels):
                if l == -1:
                    dists = [np.linalg.norm(feats_pca[i]-centroids[c]) for c in unique]
                    db_labels[i] = unique[int(np.argmin(dists))]
            try:
                from sklearn.metrics import silhouette_score
                s = silhouette_score(feats_pca, db_labels)
                if s > best_score:
                    best_score = s
                    best_labels = db_labels.copy()
            except Exception:
                pass

    # merge cluster piccoli
    final_labels = np.array(best_labels)
    unique = np.unique(final_labels)
    centroids = {c: feats_pca[final_labels==c].mean(axis=0) for c in unique}
    for c in unique:
        idxs = np.where(final_labels==c)[0]
        if len(idxs) < min_cluster_size and len(unique)>1:
            other = [oc for oc in unique if oc!=c]
            for idx in idxs:
                dists = [np.linalg.norm(feats_pca[idx]-centroids[oc]) for oc in other]
                final_labels[idx] = other[int(np.argmin(dists))]
    return final_labels, float(best_score)

# =========================
# INTERNAL CLUSTERING PER CLASSE
# =========================
def internal_clustering(features_dict, max_k_clusters=5, min_cluster_size=3):
    final_results = []
    for cls, img_feats_list in features_dict.items():
        feats_matrix = np.array([f[1] for f in img_feats_list])
        img_paths = [f[0] for f in img_feats_list]

        if len(img_feats_list) < 3 or np.allclose(np.std(feats_matrix),0):
            cluster_labels = np.zeros(len(img_feats_list), dtype=int)
            score = 0.0
        else:
            cluster_labels, score = consensus_kmeans(feats_matrix, max_k_clusters)

        id_map = {}
        for i, path in enumerate(img_paths):
            file_id = Path(path).stem.split("_")[0].lstrip("0") or "0"
            if file_id in id_map:
                cluster_labels[i] = id_map[file_id]
            else:
                id_map[file_id] = cluster_labels[i]

        for path, lbl in zip(img_paths, cluster_labels):
            final_results.append((Path(path).name, str(path), cls, int(lbl)))

    return final_results
