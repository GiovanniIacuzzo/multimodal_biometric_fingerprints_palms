import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def consensus_kmeans(feats_matrix, max_k=5, n_repeats=5, min_cluster_size=3):
    """
    Consensus KMeans clustering with PCA reduction, silhouette evaluation, 
    and small cluster merging.

    Parameters:
        feats_matrix (array): feature vectors (samples x features)
        max_k (int): maximum number of clusters to test
        n_repeats (int): number of consensus runs per k
        min_cluster_size (int): threshold to merge small clusters

    Returns:
        best_labels (array): cluster labels for each sample
        best_score (float): silhouette score of the chosen clustering
    """
    # Standardizzazione e PCA
    feats_matrix = StandardScaler().fit_transform(feats_matrix)
    n_samples, n_features = feats_matrix.shape
    n_pca = min(50, n_samples, n_features)
    feats_matrix = PCA(n_components=n_pca).fit_transform(feats_matrix)

    best_score = -1
    best_labels = None

    # Test di diversi k
    for k in range(2, min(max_k+1, n_samples)):
        all_labels = []
        for seed in range(n_repeats):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            all_labels.append(kmeans.fit_predict(feats_matrix))
        all_labels = np.array(all_labels)

        # Consensus: votazione per ogni campione
        labels = np.array([np.bincount(all_labels[:,i]).argmax() for i in range(n_samples)])
        score = silhouette_score(feats_matrix, labels)
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    # Merge piccoli cluster
    best_labels = np.array(best_labels)
    unique_clusters = np.unique(best_labels)
    cluster_centroids = {c: feats_matrix[best_labels==c].mean(axis=0) for c in unique_clusters}

    for c in unique_clusters:
        idxs = np.where(best_labels == c)[0]
        if len(idxs) < min_cluster_size:
            for idx in idxs:
                # Trova cluster più vicino tra quelli grandi
                other_clusters = [oc for oc in unique_clusters if oc != c]
                distances = [np.linalg.norm(feats_matrix[idx] - cluster_centroids[oc]) for oc in other_clusters]
                best_labels[idx] = other_clusters[np.argmin(distances)]

    return best_labels, best_score
