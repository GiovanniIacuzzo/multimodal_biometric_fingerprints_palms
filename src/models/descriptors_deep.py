"""
Modulo per l'estrazione di descrittori deep-learning da immagini di impronte digitali.
Offre un modello flessibile e robusto basato su backbone CNN/Transformer,
e un training basato su metric learning (Triplet Loss o Cosine Loss).
"""

import os
import time
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from torch import amp
from src.models.dataset import FingerprintDataset
from src.models.FeatureExtractor import FingerprintFeatureExtractor


# =====================================================
# TRAINING FUNCTION
# =====================================================
def train_deep_descriptor(
    dataset_dir,
    save_path,
    epochs=10,
    batch_size=16,
    embedding_dim=256,
    lr=1e-4,
    backbone="resnet50",
    device=None,
    pretrained=True,
    use_amp=True
):
    """
    Addestra un estrattore di feature biometrico usando triplet loss.
    """

    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🖥 Device: {device}")

    # Dataset + Dataloader
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    dataset = FingerprintDataset(dataset_dir, transform=transform, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Modello, loss e ottimizzatore
    model = FingerprintFeatureExtractor(embedding_dim, backbone, pretrained).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scaler = amp.GradScaler(device=device, enabled=(use_amp and torch.cuda.is_available()))

    print(f"📦 Backbone: {backbone} | Epochs: {epochs} | Batch size: {batch_size}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        # Barra di avanzamento
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"🧠 Epoch {epoch+1}/{epochs}")

        for batch_idx, (a, p, n) in progress_bar:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()

            with amp.autocast(device_type=device, enabled=(use_amp and device != "cpu")):
                emb_a, emb_p, emb_n = model(a), model(p), model(n)
                loss = criterion(emb_a, emb_p, emb_n)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        print(f"✅ Epoch {epoch+1}/{epochs} completato - Loss media: {avg_loss:.6f} - Tempo: {time.time()-start:.1f}s")

        # Salvataggio checkpoint
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint salvato in {save_path}")

    print("🏁 Training completato.")
    return model


# =====================================================
# MAIN ENTRYPOINT (eseguibile da pipeline)
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Train deep descriptor model for fingerprint recognition")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path alla cartella del dataset di impronte (organizzato per soggetto)")
    parser.add_argument("--save_path", type=str, default="data/models/deep_descriptor.pth",
                        help="Percorso file per salvare il modello addestrato")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "vit_b_16", "efficientnet_b0"],
                        help="Backbone del feature extractor")
    parser.add_argument("--no_pretrained", action="store_true", help="Disabilita pesi pre-addestrati")
    parser.add_argument("--no_amp", action="store_true", help="Disabilita mixed precision (AMP)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    train_deep_descriptor(
        dataset_dir=args.dataset_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        use_amp=not args.no_amp
    )


if __name__ == "__main__":
    main()
