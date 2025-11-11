import os
import time
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import math

from classifier.dataset2.dataset import FingerprintDataset
from classifier.models.ssl_model import SSLModel
from classifier.utils.loss import NTXentLoss
from classifier.utils.utils import save_model, load_model

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CosineWarmupScheduler:
    """Simple cosine lr with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * float(epoch+1) / float(max(1, self.warmup_epochs))
        else:
            t = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * t))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

def train_ssl(
    model: torch.nn.Module = None,
    dataloader: DataLoader = None,
    data_dir: str = None,
    device: torch.device = None,
    epochs: int = 50,
    lr: float = 3e-4,
    temperature: float = 0.5,
    save_dir: str = "save_models",
    pretrained_backbone: bool = True,
    batch_size: int = 64,
    num_workers: int = 4,
    gradient_clip: float = 1.0,
    amp: bool = True,
    seed: int = 42,
    checkpoint_path: str = None,
    save_every: int = 5,
    warmup_epochs: int = 5
):
    """
    Robust SSL training loop (NT-Xent).
    Returns trained model.
    """
    device = device or default_device()
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # create dataloader if needed
    if dataloader is None:
        if data_dir is None:
            raise ValueError("If dataloader is not provided, data_dir must be given.")
        dataset = FingerprintDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, pin_memory=True)

    # create model if needed
    if model is None:
        model = SSLModel(backbone_name="resnet18", pretrained=pretrained_backbone).to(device)
    else:
        model = model.to(device)

    # load checkpoint if provided (resume)
    start_epoch = 0
    best_loss = float("inf")
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = load_model(checkpoint_path)
        if isinstance(ckpt, dict) and "epoch" in ckpt and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", best_loss)
            print(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}")

    criterion = NTXentLoss(batch_size=batch_size, temperature=temperature, device=str(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs, base_lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="b"):
            # batch expected as (x_i, x_j)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_i, x_j = batch
            else:
                raise RuntimeError("FingerprintDataset should return (x_i, x_j) for SSL training")

            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                z_i = model(x_i)
                z_j = model(x_j)
                loss = criterion(z_i, z_j)

            scaler.scale(loss).backward()

            # gradient clipping
            if gradient_clip is not None and gradient_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step(epoch)
        avg_loss = epoch_loss / max(1, num_batches)
        t1 = time.time()
        print(f"[Epoch {epoch+1}/{epochs}] loss={avg_loss:.4f} time={t1-t0:.1f}s lr={optimizer.param_groups[0]['lr']:.2e}")

        # save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"ssl_epoch{epoch+1}.pth")
            save_model({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss
            }, ckpt_path)

        # update best & save
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "ssl_model_best.pth")
            save_model({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_loss": best_loss
            }, best_path)

    # final save
    final_path = os.path.join(save_dir, "ssl_model_final.pth")
    save_model({
        "epoch": epochs - 1,
        "model_state": model.state_dict(),
        "best_loss": best_loss
    }, final_path)
    print("SSL training completed. Final model saved to:", final_path)
    return model
