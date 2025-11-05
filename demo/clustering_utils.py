# demo/clustering_utils.py
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.metrics import silhouette_score

# =====================================================
# STANDARDIZZAZIONE
# =====================================================
def standardize_global_features(features_dict):
    """Applica StandardScaler globale in modo robusto."""
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
            scaled = scaler.transform(padded.reshape(1, -1)).flatten()
            features_dict[cls][i] = (path, scaled[:feat.shape[0]])
    return features_dict, scaler

# =====================================================
# KMEANS ROBUSTO + FALLBACK DBSCAN
# =====================================================
def consensus_kmeans(feats_matrix, max_k=5, n_repeats=5, min_cluster_size=3):
    n_samples = feats_matrix.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int), 0.0

    scaler = StandardScaler()
    feats_std = scaler.fit_transform(feats_matrix)

    n_pca = min(30, feats_std.shape[1], n_samples)
    if n_pca < 1:
        n_pca = 1
    feats_pca = PCA(n_components=n_pca).fit_transform(feats_std)

    best_score = -1.0
    best_labels = np.zeros(n_samples, dtype=int)
    max_k_test = min(max_k, n_samples - 1)

    for k in range(2, max_k_test + 1):
        all_labels = []
        for seed in range(n_repeats):
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            lbl = km.fit_predict(feats_pca)
            all_labels.append(lbl)
        all_labels = np.array(all_labels)
        labels = np.array([np.bincount(all_labels[:, i]).argmax() for i in range(n_samples)])
        try:
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

    # merge cluster piccoli
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

# =====================================================
# SOGLIE ADATTIVE
# =====================================================
def estimate_adaptive_thresholds(std_angles, percentiles=(33, 66)):
    t1 = np.percentile(std_angles, percentiles[0])
    t2 = np.percentile(std_angles, percentiles[1])
    return t1, t2

# =====================================================
# ASSEGNAZIONE CLASSE GLOBALE
# =====================================================
def assign_global_label(std_angle, thresholds):
    """
    Assegna una classe globale in base allo std_angle e alle soglie adattive.
    Output: 'Arch', 'Loop' o 'Whorl'
    """
    t1, t2 = thresholds
    if std_angle < t1:
        return "Arch"
    elif std_angle < t2:
        return "Loop"
    else:
        return "Whorl"

# =====================================================
# CLUSTERING INTERNO PER OGNI CLASSE
# =====================================================

def internal_clustering(features_dict, max_k_clusters=5):
    results = []

    for global_class, img_feats_list in features_dict.items():
        if not img_feats_list:
            continue

        # === Raggruppa per ID ===
        id_groups = {}
        for path, feat in img_feats_list:
            path = Path(path)
            fname = path.stem

            # L'ID è tutto ciò che precede il primo underscore
            file_id = fname.split("_")[0].lstrip("0") or "0"

            # Aggiunge alla lista di immagini per quell'ID
            id_groups.setdefault(file_id, []).append((path, feat))

        # === Calcola il vettore medio per ciascun ID ===
        id_features = []
        id_keys = []
        for file_id, items in id_groups.items():
            feats = np.array([f for _, f in items if f is not None])
            if feats.size == 0:
                continue
            mean_feat = feats.mean(axis=0)
            id_features.append(mean_feat)
            id_keys.append(file_id)

        id_features = np.array(id_features, dtype=np.float32)
        n_ids = len(id_keys)

        if n_ids == 0:
            print(f"⚠️ Nessuna feature valida per {global_class}")
            continue

        # === Clustering a livello di ID ===
        if n_ids == 1:
            id_labels = np.zeros(n_ids, dtype=int)
            score = 0.0
        else:
            try:
                id_labels, score = consensus_kmeans(id_features, max_k=max_k_clusters)
            except Exception as e:
                print(f"⚠️ Errore in consensus_kmeans per {global_class}: {e}")
                id_labels = np.zeros(n_ids, dtype=int)
                score = 0.0

        print(f"📊 {global_class}: {len(np.unique(id_labels))} cluster "
              f"su {n_ids} ID (silhouette={score:.3f})")

        # === Assegna la label di cluster a tutte le immagini di quell’ID ===
        for lbl, file_id in zip(id_labels, id_keys):
            for path, _ in id_groups[file_id]:
                results.append([
                    path.name,        # filename
                    str(path),        # percorso completo
                    global_class,     # classe globale (Arch/Loop/Whorl)
                    int(lbl)          # cluster locale
                ])

    return results
