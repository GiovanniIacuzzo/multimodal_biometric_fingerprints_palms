import numpy as np
from pathlib import Path
from demo.clustering import consensus_kmeans
from sklearn.metrics import silhouette_score

# ============================================================
# GLOBAL CLASS ASSIGNMENT
# ============================================================
def assign_global_label(std_angle, adaptive_thresholds=None):
    """
    Assegna la classe globale in base alla deviazione standard
    dell'orientazione (in gradi).

    - Usa soglie fisse o adattive (calcolate su dataset).
    - Restituisce una delle tre classi: Arch, Loop, Whorl.
    """
    if adaptive_thresholds is not None:
        t1, t2 = adaptive_thresholds
    else:
        # Soglie empiriche di default
        t1, t2 = 20, 45

    if std_angle < t1:
        return "Arch"
    elif std_angle < t2:
        return "Loop"
    else:
        return "Whorl"


def estimate_adaptive_thresholds(std_angles, percentiles=(33, 66)):
    """
    Calcola soglie adattive dai valori reali di deviazione standard
    delle immagini nel dataset.
    """
    t1 = np.percentile(std_angles, percentiles[0])
    t2 = np.percentile(std_angles, percentiles[1])
    return float(t1), float(t2)


# ============================================================
# INTERNAL CLUSTERING
# ============================================================
def internal_clustering(features_dict, max_k_clusters=5, min_cluster_size=3):
    """
    Esegue un clustering interno per ciascuna classe di immagini usando consensus KMeans.
    Gestisce automaticamente casi di bassa varianza o dataset ridotti.

    Parametri:
    ----------
    features_dict : dict
        Dizionario {classe: [(path, features), ...]}.
    max_k_clusters : int
        Numero massimo di cluster da testare per classe.
    min_cluster_size : int
        Soglia minima di elementi per mantenere un cluster (altrimenti viene fuso con il più vicino).

    Restituisce:
    ------------
    final_results : list of tuples
        [(nome_file, path_assoluto, classe, etichetta_cluster)]
    """
    final_results = []

    for cls, img_feats_list in features_dict.items():
        # Estraggo feature e path
        feats_matrix = np.array([f[1] for f in img_feats_list])
        img_paths = [f[0] for f in img_feats_list]

        # Controllo che ci siano abbastanza campioni per il clustering
        if len(img_feats_list) < 3 or np.allclose(np.std(feats_matrix), 0):
            print(f"Classe '{cls}' ha varianza nulla o pochi campioni ({len(img_feats_list)}) → assegnato cluster unico.")
            cluster_labels = np.zeros(len(img_feats_list), dtype=int)
            score = 0.0
        else:
            # Applica consensus KMeans robusto
            cluster_labels, score = consensus_kmeans(feats_matrix, max_k_clusters)
            unique_labels = np.unique(cluster_labels)
            print(f"Classe '{cls}': {len(unique_labels)} cluster trovati, silhouette = {score:.2f}")

            # Fusione opzionale dei cluster troppo piccoli
            counts = np.bincount(cluster_labels)
            small_clusters = np.where(counts < min_cluster_size)[0]

            if len(small_clusters) > 0:
                print(f"Fusione di {len(small_clusters)} piccoli cluster per la classe '{cls}'...")
                for sc in small_clusters:
                    idxs = np.where(cluster_labels == sc)[0]
                    for idx in idxs:
                        # Calcola distanza media verso i cluster validi
                        distances = []
                        for c in unique_labels:
                            if c == sc or np.sum(cluster_labels == c) == 0:
                                continue
                            centroid_c = feats_matrix[cluster_labels == c].mean(axis=0)
                            d = np.linalg.norm(feats_matrix[idx] - centroid_c)
                            distances.append((d, c))
                        # Riassegna al cluster più vicino
                        if distances:
                            cluster_labels[idx] = min(distances, key=lambda x: x[0])[1]

        # Aggiunge risultati finali
        for path, lbl in zip(img_paths, cluster_labels):
            final_results.append((Path(path).name, str(path), cls, int(lbl)))

    print("\nClustering interno completato con successo per tutte le classi.\n")
    return final_results