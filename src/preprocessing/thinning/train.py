import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from src.preprocessing.thinning.dataset import FingerprintThinningDataset
from src.preprocessing.thinning.model import UNet

# ======================================
# LOSS FUNCS
# ======================================
def dice_loss(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + dice_loss(pred, target)


# ======================================
# CONFIG
# ======================================
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-3
IMG_SIZE = (256, 256)
AUGMENT = True
VAL_SPLIT = 0.2
NUM_WORKERS = 0

BINARY_DIR = "dataset/processed/debug/binary"
THINNING_DIR = "dataset/processed/debug/skeleton"
SAVE_DIR = "src/preprocessing/thinning/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================
# DATASET
# ======================================
dataset = FingerprintThinningDataset(
    binary_dir=BINARY_DIR,
    thinning_dir=THINNING_DIR,
    img_size=IMG_SIZE,
    augment=AUGMENT
)

val_len = int(len(dataset) * VAL_SPLIT)
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")


# ======================================
# MODEL
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_ch=1, out_ch=1).to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ======================================
# TRAIN LOOP
# ======================================
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]"):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # VALIDATION
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Salvataggio miglior modello
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Nuovo miglior modello salvato in {save_path}")

print("Training completato!")
