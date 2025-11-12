import os
import time
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import math
import numpy as np

from classifier.dataset2.dataset import FingerprintDataset
from classifier.models.ssl_model import SSLModel
from classifier.utils.loss import NTXentLoss
from classifier.utils.utils import save_model, load_model


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CosineWarmupScheduler:
    """Cosine LR with warmup."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            t = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * t))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


def train_ssl(
    model=None,
    dataloader=None,
    data_dir=None,
    device=None,
    epochs=100,
    lr=3e-4,
    temperature=0.5,
    save_dir="save_models",
    pretrained_backbone=True,
    batch_size=64,
    num_workers=4,
    gradient_clip=1.0,
    amp=True,
    seed=42,
    checkpoint_path=None,
    save_every=10,
    warmup_epochs=5,
    early_stop_patience=15,
):
    """
    SSL training loop compatibile con CUDA / MPS / CPU.
    """
    device = device or default_device()
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    if dataloader is None:
        if not data_dir:
            raise ValueError("Either dataloader or data_dir must be provided.")
        dataset = FingerprintDataset(data_dir)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True
        )

    model = model or SSLModel(backbone_name="resnet18", pretrained=pretrained_backbone)
    model = model.to(device)

    criterion = NTXentLoss(batch_size=batch_size, temperature=temperature, device=str(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs, epochs, lr)

    # Mixed precision solo su CUDA
    use_amp = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss, patience_counter = float("inf"), 0
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = []
        t0 = time.time()

        for x_i, x_j in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="b"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    z_i, z_j = model(x_i), model(x_j)
                    loss = criterion(z_i, z_j)
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_i, z_j = model(x_i), model(x_j)
                loss = criterion(z_i, z_j)
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        lr_now = scheduler.step(epoch)
        print(f"[{epoch+1}/{epochs}] loss={avg_loss:.4f} lr={lr_now:.2e} time={time.time()-t0:.1f}s")

        # Early stopping & salvataggio best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_loss": best_loss},
                os.path.join(save_dir, "ssl_best.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[STOP] Early stopping at epoch {epoch+1}")
                break

        # Salvataggio checkpoint intermedio
        if (epoch + 1) % save_every == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(save_dir, f"ssl_epoch{epoch+1}.pth")
            )

    print(f"[DONE] Training completed. Best loss: {best_loss:.4f}")
    return model
