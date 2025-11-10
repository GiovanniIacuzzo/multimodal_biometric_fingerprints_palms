import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from classifier.dataset2.dataset import BaseDataset
from classifier.models.backbone import CNNBackbone
from classifier.utils.utils import save_model, load_model
from classifier.utils.extract_embeddings import extract_embeddings
from classifier.utils.cluster_embeddings import cluster_kmeans, cluster_hdbscan, visualize_tsne, visualize_umap
from classifier.config import CONFIG

def main():
    # ============================================================
    # CONFIGURAZIONE
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)

    # ============================================================
    # 1) Dataset e split
    # ============================================================
    dataset = BaseDataset(CONFIG["dataset_path"])
    train_len = int(len(dataset) * CONFIG["train_split"])
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # ============================================================
    # 2) Modello backbone + classificatore
    # ============================================================
    backbone = CNNBackbone(model_name=CONFIG["backbone"]).to(device)

    # Carico i pesi pre-addestrati SSL
    checkpoint_path = os.path.join(CONFIG["save_dir"], "ssl_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Se il checkpoint contiene "state_dict", usiamolo
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Mapping chiavi: aggiungi prefisso 'backbone.' se necessario e ignora projection head
    backbone_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("projection_head"):
            continue
        if not k.startswith("backbone."):
            backbone_state_dict[f"backbone.{k}"] = v
        else:
            backbone_state_dict[k] = v

    # Carica i pesi nel backbone (strict=False per differenze eventuali)
    backbone.load_state_dict(backbone_state_dict, strict=False)

    classifier = nn.Linear(backbone.output_dim, CONFIG["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=CONFIG["lr"])

    # ============================================================
    # 3) Training supervisionato
    # ============================================================
    for epoch in range(CONFIG["finetune_epochs"]):
        backbone.train()
        classifier.train()
        total_loss = 0
        correct = 0

        for batch in train_loader:
            # Estrazione imgs e labels in modo sicuro
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, labels = batch
            else:
                raise ValueError(f"Batch inaspettato: {batch}")

            # Se imgs è ancora lista/tuple di tensori, trasformala in tensor unico
            if isinstance(imgs, (list, tuple)):
                imgs = torch.stack(imgs)

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = backbone(imgs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{CONFIG['finetune_epochs']}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    print("Fine-tuning completed.")

    # ============================================================
    # 4) Salvataggio modello
    # ============================================================
    save_model({
        "backbone": backbone.state_dict(),
        "classifier": classifier.state_dict()
    }, os.path.join(CONFIG["save_dir"], "finetuned_classifier.pth"))

    # ============================================================
    # 5) Estrazione embeddings dal modello fine-tuned
    # ============================================================
    embeddings, filenames = extract_embeddings(
        data_dir=CONFIG["dataset_path"],
        model=backbone,  # usa solo backbone per embeddings
        device=device,
        batch_size=CONFIG["batch_size"]
    )

    # Salvataggio embeddings
    torch.save({
        "embeddings": embeddings,
        "filenames": filenames
    }, os.path.join(CONFIG["save_dir"], "embeddings_finetuned.pth"))

    # ============================================================
    # 6) Clustering e visualizzazione
    # ============================================================
    labels_kmeans, _ = cluster_kmeans(embeddings, n_clusters=CONFIG["n_clusters"])
    labels_hdbscan, _ = cluster_hdbscan(embeddings, min_cluster_size=CONFIG["min_cluster_size"])

    visualize_tsne(embeddings, labels_kmeans, save_path=os.path.join(CONFIG["figures_dir"], "tsne_kmeans_finetuned.png"))
    visualize_umap(embeddings, labels_kmeans, save_path=os.path.join(CONFIG["figures_dir"], "umap_kmeans_finetuned.png"))
    visualize_tsne(embeddings, labels_hdbscan, save_path=os.path.join(CONFIG["figures_dir"], "tsne_hdbscan_finetuned.png"))
    visualize_umap(embeddings, labels_hdbscan, save_path=os.path.join(CONFIG["figures_dir"], "umap_hdbscan_finetuned.png"))

    print("Supervised fine-tuning pipeline completed.")


if __name__ == "__main__":
    main()
