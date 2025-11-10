import os
import torch
import csv
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from classifier.dataset2.dataset import BaseDataset, FingerprintDataset
from classifier.models.ssl_model import SSLModel
from classifier.utils.train_ssl import train_ssl
from classifier.utils.extract_embeddings import extract_embeddings
from classifier.utils.cluster_embeddings import cluster_kmeans, cluster_hdbscan, visualize_tsne, visualize_umap
from config.config_classifier import CONFIG

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)

    # ---------------------------
    # 1) Dataset SSL e DataLoader
    # ---------------------------
    dataset_ssl = FingerprintDataset(CONFIG["dataset_path"])
    dataloader_ssl = DataLoader(
        dataset_ssl,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # ---------------------------
    # 2) Modello SSL
    # ---------------------------
    model = SSLModel(
        backbone_name=CONFIG["backbone"],
        pretrained=True,
        embedding_dim=512,
        proj_hidden_dim=512,
        proj_output_dim=128,
        proj_num_layers=2,
        freeze_backbone=False
    ).to(device)

    # Training contrastive learning
    model = train_ssl(
        model=model,
        dataloader=dataloader_ssl,
        data_dir=CONFIG["dataset_path"],
        device=device,
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        temperature=CONFIG["temperature"],
        save_path=CONFIG["save_dir"]
    )

    # ---------------------------
    # 3) Estrazione embeddings
    # ---------------------------
    dataset_emb = BaseDataset(CONFIG["dataset_path"])
    embeddings, filenames = extract_embeddings(
        data_dir=CONFIG["dataset_path"],
        model=model,
        device=device,
        batch_size=CONFIG["batch_size"]
    )

    torch.save({"embeddings": embeddings, "filenames": filenames},
               os.path.join(CONFIG["save_dir"], "embeddings.pth"))

    # ---------------------------
    # 4) Clustering globale
    # ---------------------------
    labels_kmeans, _ = cluster_kmeans(embeddings, n_clusters=CONFIG["n_clusters"])
    labels_hdbscan, _ = cluster_hdbscan(embeddings, min_cluster_size=CONFIG["min_cluster_size"])

    visualize_tsne(embeddings, labels_kmeans,
                   save_path=os.path.join(CONFIG["figures_dir"], "tsne_kmeans.png"))
    visualize_umap(embeddings, labels_kmeans,
                   save_path=os.path.join(CONFIG["figures_dir"], "umap_kmeans.png"))
    visualize_tsne(embeddings, labels_hdbscan,
                   save_path=os.path.join(CONFIG["figures_dir"], "tsne_hdbscan.png"))
    visualize_umap(embeddings, labels_hdbscan,
                   save_path=os.path.join(CONFIG["figures_dir"], "umap_hdbscan.png"))

    # ---------------------------
    # 5) Raggruppa embeddings per ID
    # ---------------------------
    id_to_embeddings = defaultdict(list)
    id_to_filenames = defaultdict(list)
    for emb, fname in zip(embeddings, filenames):
        # Estrai ID dai nomi dei file (modifica secondo convenzione)
        file_id = fname.split("_")[0].lstrip("0") or fname.split("_")[0]
        id_to_embeddings[file_id].append(emb)
        id_to_filenames[file_id].append(fname)

    # Aggrega embeddings per ID (media)
    id_list = []
    agg_embeddings = []
    for fid, embs in id_to_embeddings.items():
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        id_list.append(fid)
        agg_embeddings.append(mean_emb)
    agg_embeddings = np.stack(agg_embeddings, axis=0)

    # ---------------------------
    # 6) Clustering a livello ID
    # ---------------------------
    id_labels, _ = cluster_kmeans(agg_embeddings, n_clusters=CONFIG["n_clusters"])

    # ---------------------------
    # 7) Espandi cluster ID a immagini e salva CSV
    # ---------------------------
    csv_path = os.path.join(CONFIG["save_dir"], "id_level_clusters.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "global_class", "cluster_in_class"])
        for fid, cluster in zip(id_list, id_labels):
            for fname in id_to_filenames[fid]:
                writer.writerow([fname,
                                 os.path.join(CONFIG["dataset_path"], fname),
                                 fid,           # global_class = ID
                                 int(cluster)]) # cluster_in_class

    print(f"ID-level clustering and CSV saved at {csv_path}")
    print("SSL pipeline completed.")

if __name__ == "__main__":
    main()
