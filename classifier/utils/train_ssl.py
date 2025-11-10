import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from classifier.dataset2.dataset import FingerprintDataset
from classifier.models.backbone import CNNBackbone
from classifier.models.projection_head import ProjectionHead
from classifier.models.ssl_model import SSLModel
from classifier.utils.loss import NTXentLoss
from classifier.utils.utils import save_model


def train_ssl(
    model=None,
    dataloader=None,
    device=None,
    epochs=50,
    lr=3e-4,
    temperature=0.5,
    save_path="save_model/ssl_model.pt",
    pretrained_backbone=True,
    data_dir=None,
    batch_size=64
):
    """
    Funzione per addestramento self-supervised contrastive learning (NT-Xent).
    """

    device = device or ("cuda" if torch.cuda.is_available() else "mps")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ----------------------------
    # Se non viene passato il modello, crealo
    # ----------------------------
    if model is None:
        backbone = CNNBackbone(pretrained=pretrained_backbone)
        projection = ProjectionHead(input_dim=backbone.output_dim, hidden_dim=512, output_dim=128)
        model = SSLModel(backbone, projection).to(device)

    # ----------------------------
    # Se non viene passato il dataloader, crealo
    # ----------------------------
    if dataloader is None:
        if data_dir is None:
            raise ValueError("Se dataloader non passato, serve data_dir")
        dataset = FingerprintDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    # ----------------------------
    # LOSS & OPTIMIZER
    # ----------------------------
    criterion = NTXentLoss(batch_size=batch_size, temperature=temperature, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_i, x_j in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)

            optimizer.zero_grad()
            z_i = model(x_i)
            z_j = model(x_j)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        # Salvataggio checkpoint ogni 5 epoche
        if (epoch + 1) % 5 == 0:
            save_model(model, os.path.join(os.path.dirname(save_path), f"ssl_model_epoch{epoch+1}.pth"))
    save_model_ssl = save_path + "/model_ssl.pt"
    print("Training SSL completato!")
    save_model(model, save_model_ssl)
    return model
