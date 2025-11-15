import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocessing.binarize.model import UNetSmall
from src.preprocessing.binarize.dataset import FingerprintBinaryDataset


def load_config(path="config/config_binarize.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for img_in, img_gt in tqdm(loader, desc="Training"):
        img_in = img_in.to(device)
        img_gt = img_gt.to(device)

        # Ground truth è 0-1 → BCEWithLogitsLoss richiede float
        img_gt = (img_gt > 0.5).float()

        optimizer.zero_grad()
        pred = model(img_in)
        loss = criterion(pred, img_gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for img_in, img_gt in tqdm(loader, desc="Validation"):
            img_in = img_in.to(device)
            img_gt = img_gt.to(device)
            img_gt = (img_gt > 0.5).float()

            pred = model(img_in)
            loss = criterion(pred, img_gt)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    cfg = load_config()

    device = cfg["training"]["device"]
    print(f"Using device: {device}")

    # --- Dataset ---
    train_ds = FingerprintBinaryDataset(
        root_dir=cfg["paths"]["train_dir"],
        split="train",
        val_split=cfg["paths"]["val_split"],
        subset_size=cfg["dataset"]["subset_size"],
        subset_seed=cfg["dataset"]["subset_seed"],
        transform=None
    )

    val_ds = FingerprintBinaryDataset(
        root_dir=cfg["paths"]["train_dir"],
        split="val",
        val_split=cfg["paths"]["val_split"],
        subset_size=cfg["dataset"]["subset_size"],
        subset_seed=cfg["dataset"]["subset_seed"],
        transform=None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # --- Model ---
    model = UNetSmall(in_channels=1, out_channels=1).to(device)

    # Loss adeguata per mappe binarie
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    # --- Training loop ---
    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("data/checkpoints/binarize", exist_ok=True)
            torch.save(model.state_dict(), "data/checkpoints/binarize/best_unet_small.pth")
            print("✓ Saved best model")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
