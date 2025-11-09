import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -------------------------
# Utility thresholds sugli std_angle
# -------------------------
def estimate_adaptive_thresholds(std_angles, percentiles=(33, 66)):
    t1 = float(np.percentile(std_angles, percentiles[0]))
    t2 = float(np.percentile(std_angles, percentiles[1]))
    return t1, t2

def assign_global_label(std_angle, thresholds=None):
    """
    Assegna una delle tre classi globali basata su std_angle (in gradi).
    thresholds: (t1, t2) oppure None -> default empirico
    """
    if thresholds is None:
        t1, t2 = 20.0, 45.0
    else:
        t1, t2 = thresholds
    if std_angle < t1:
        return "Arch"
    elif std_angle < t2:
        return "Loop"
    else:
        return "Whorl"

# -------------------------
# Standardizzazione globale (su vettori padded)
# -------------------------
def standardize_global_features(features_dict):
    """
    features_dict: {class: [(id, feat_vector), ...]}
    Restituisce (features_dict_scaled, scaler). I vettori sono scalati e mantenuti nelle stesse dimensioni.
    """
    all_feats = []
    keys = []
    for cls, lst in features_dict.items():
        for _id, feat in lst:
            all_feats.append(feat)
            keys.append((cls, _id))

    if not all_feats:
        return features_dict, None

    max_len = max(f.shape[0] for f in all_feats)
    stacked = np.zeros((len(all_feats), max_len), dtype=np.float32)
    for i, f in enumerate(all_feats):
        stacked[i, : f.shape[0]] = f

    if stacked.shape[0] < 2:
        return features_dict, None

    scaler = StandardScaler().fit(stacked)
    transformed = scaler.transform(stacked)

    # rimappa i vettori scalati alla struttura originale (tagliando il padding)
    idx = 0
    for cls, lst in features_dict.items():
        for j in range(len(lst)):
            _id, feat = lst[j]
            orig_len = feat.shape[0]
            scaled = transformed[idx, :orig_len].astype(np.float32)
            features_dict[cls][j] = (_id, scaled)
            idx += 1

    return features_dict, scaler

# -------------------------
# consensus_kmeans robusto con fallback DBSCAN
# -------------------------
def consensus_kmeans(feats_matrix, max_k=5, n_repeats=5, min_cluster_size=3):
    """
    feats_matrix: numpy array (n_samples, n_features)
    Ritorna: labels (n_samples,), best_silhouette_score float
    """
    n_samples = feats_matrix.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int), 0.0

    # Standardize (interno)
    scaler = StandardScaler()
    feats_std = scaler.fit_transform(feats_matrix)

    # PCA per ridurre dimensionalità (ma manteniamo quanto possibile)
    n_pca = min(30, feats_std.shape[1], n_samples)
    if n_pca < 1:
        n_pca = 1
    feats_pca = PCA(n_components=n_pca).fit_transform(feats_std)

    best_score = -1.0
    best_labels = np.zeros(n_samples, dtype=int)
    max_k_test = min(max_k, max(2, n_samples - 1))

    for k in range(2, max_k_test + 1):
        all_labels = []
        for seed in range(n_repeats):
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            lbl = km.fit_predict(feats_pca)
            all_labels.append(lbl)
        all_labels = np.array(all_labels)
        # consensus via majority vote per sample
        labels = np.array([np.bincount(all_labels[:, i]).argmax() for i in range(n_samples)])
        try:
            score = silhouette_score(feats_pca, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    # fallback DBSCAN se silhouette debole
    if best_score < 0.25:
        db = DBSCAN(eps=0.8, min_samples=3).fit(feats_pca)
        db_labels = db.labels_
        if len(set(db_labels)) > 1:
            unique = [c for c in set(db_labels) if c != -1]
            if unique:
                centroids = {c: feats_pca[db_labels == c].mean(axis=0) for c in unique}
                for i, l in enumerate(db_labels):
                    if l == -1:
                        dists = [np.linalg.norm(feats_pca[i] - centroids[c]) for c in unique]
                        db_labels[i] = unique[int(np.argmin(dists))]
                try:
                    s = silhouette_score(feats_pca, db_labels)
                    if s > best_score:
                        best_score = s
                        best_labels = db_labels.copy()
                except Exception:
                    pass

    # Merge cluster troppo piccoli (min_cluster_size)
    final_labels = np.array(best_labels)
    unique = np.unique(final_labels)
    centroids = {c: feats_pca[final_labels == c].mean(axis=0) for c in unique}
    for c in unique:
        idxs = np.where(final_labels == c)[0]
        if len(idxs) < min_cluster_size and len(unique) > 1:
            other = [oc for oc in unique if oc != c]
            for idx in idxs:
                dists = [np.linalg.norm(feats_pca[idx] - centroids[oc]) for oc in other]
                final_labels[idx] = other[int(np.argmin(dists))]

    return final_labels, float(best_score)

# -------------------------
# internal_clustering: ora lavora a livello ID
# -------------------------
def internal_clustering(features_dict, max_k_clusters=5, min_cluster_size=3):
    """
    features_dict: {class: [(id, feat), ...]}
    Restituisce lista di tuples: [(id, class, cluster_label), ...]
    """
    results = []

    for global_class, id_feats_list in features_dict.items():
        if not id_feats_list:
            continue

        ids = [t[0] for t in id_feats_list]
        feats = np.array([t[1] for t in id_feats_list], dtype=np.float32)
        n_ids = len(ids)

        if n_ids == 1:
            id_labels = np.zeros(n_ids, dtype=int)
            score = 0.0
        else:
            try:
                id_labels, score = consensus_kmeans(feats, max_k=max_k_clusters, min_cluster_size=min_cluster_size)
            except Exception as e:
                print(f"Errore in consensus_kmeans per {global_class}: {e}")
                id_labels = np.zeros(n_ids, dtype=int)
                score = 0.0

        print(f"{global_class}: {len(np.unique(id_labels))} cluster su {n_ids} ID (silhouette={score:.3f})")

        for fid, lbl in zip(ids, id_labels):
            results.append((str(fid), global_class, int(lbl)))

    return results
