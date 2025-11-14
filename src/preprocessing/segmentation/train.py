import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import yaml
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm

# Optional: albumentations for strong augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except Exception:
    HAS_ALB = False

from torch.utils.tensorboard import SummaryWriter

# Import your dataset + model (assumes package import works; run from project root)
from src.preprocessing.segmentation.dataset import FingerprintDataset
from src.preprocessing.segmentation.model import FingerprintSegmentationModel


# ---------------------------
# Utilities: losses + metrics
# ---------------------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = targets.float()
        TP = (preds * targets).sum(dim=(1, 2, 3))
        FP = (preds * (1 - targets)).sum(dim=(1, 2, 3))
        FN = ((1 - preds) * targets).sum(dim=(1, 2, 3))
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        loss = (1 - tversky) ** self.gamma
        return loss.mean()


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()


def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = (target > 0.5)
    intersection = (pred & target).sum(dim=(1, 2, 3)).float()
    union = pred.sum(dim=(1, 2, 3)).float() + target.sum(dim=(1, 2, 3)).float()
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def iou_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = (target > 0.5)
    inter = (pred & target).sum(dim=(1, 2, 3)).float()
    union = (pred | target).sum(dim=(1, 2, 3)).float()
    return ((inter + eps) / (union + eps)).mean().item()


# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(cfg):
    img_h, img_w = cfg["dataset"]["image_size"]
    if cfg["augmentation"]["use_albumentations"] and HAS_ALB:
        aug = [
            A.Resize(img_h, img_w),
        ]
        if cfg["augmentation"]["horizontal_flip_p"] > 0:
            aug.append(A.HorizontalFlip(p=cfg["augmentation"]["horizontal_flip_p"]))
        ssr = cfg["augmentation"]["shift_scale_rotate"]
        aug.append(A.ShiftScaleRotate(shift_limit=ssr["shift_limit"],
                                      scale_limit=ssr["scale_limit"],
                                      rotate_limit=ssr["rotate_limit"],
                                      p=ssr["p"]))
        if cfg["augmentation"]["brightness_contrast_p"] > 0:
            aug.append(A.RandomBrightnessContrast(p=cfg["augmentation"]["brightness_contrast_p"]))
        if cfg["augmentation"]["gauss_noise_p"] > 0:
            aug.append(A.GaussNoise(p=cfg["augmentation"]["gauss_noise_p"]))
        if cfg["augmentation"]["elastic_transform_p"] > 0:
            aug.append(A.ElasticTransform(p=cfg["augmentation"]["elastic_transform_p"]))
        aug.append(ToTensorV2())
        return A.Compose(aug)
    else:
        # fallback simple torchvision-like transform via lambda in dataset (assumes dataset can accept callable)
        def simple_transform(img, mask):
            # img, mask are numpy arrays in dataset interface assumed
            import torchvision.transforms.functional as TF
            import torchvision.transforms as T
            from PIL import Image
            pil_img = Image.fromarray(img) if img.ndim == 2 else Image.fromarray(img[:, :, ::-1])
            pil_mask = Image.fromarray(mask)
            pil_img = pil_img.resize((img_w, img_h))
            pil_mask = pil_mask.resize((img_w, img_h))
            # to tensor and normalize 0..1
            img_t = T.ToTensor()(pil_img)
            mask_t = T.ToTensor()(pil_mask)
            return img_t, mask_t
        return simple_transform


def collate_fn(batch):
    # If dataset returns (img_tensor, mask_tensor) for albumentations ToTensorV2 the tensors will already be torch.Tensor
    imgs = []
    masks = []
    for item in batch:
        imgs.append(item[0])
        masks.append(item[1])
    imgs = torch.stack(imgs)
    masks = torch.stack(masks)
    return imgs, masks


# ---------------------------
# Main train function
# ---------------------------
def train_from_config(cfg_path: str):
    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set seed
    set_seed(cfg["experiment"].get("seed", 42))

    # Device selection
    dev_choice = cfg["training"]["device"].lower()
    if dev_choice == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev_choice)

    print(f"[INFO] Using device: {device}")

    # Build transforms
    transform = build_transforms(cfg)

    # Instantiate dataset
    dataset = FingerprintDataset(
        cfg["dataset"]["img_dir"],
        cfg["dataset"]["mask_dir"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        transform=transform
    )
    N = len(dataset)
    val_split = cfg["dataset"].get("val_split", 0.2)
    indices = np.arange(N)
    np.random.shuffle(indices)
    val_count = int(N * val_split)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle"],
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"].get("pin_memory", False),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        num_workers=max(0, cfg["dataset"]["num_workers"] // 2),
        pin_memory=cfg["dataset"].get("pin_memory", False),
        collate_fn=collate_fn
    )

    # Model
    model = FingerprintSegmentationModel(
        num_labels=cfg["model"]["num_labels"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        pretrained_model=cfg["model"]["pretrained_model"]
    )
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])

    # Scheduler
    if cfg["scheduler"]["type"] == "OneCycleLR":
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            raise ValueError("Train loader vuoto! Controlla dataset e batch_size.")

        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg["optimizer"]["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=cfg["training"]["epochs"],
            pct_start=cfg["scheduler"].get("pct_start", 0.1),
            div_factor=cfg["scheduler"].get("div_factor", 25.0),
        )
        scheduler_type = "onecycle"
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=cfg["scheduler"].get("factor", 0.5),
                                      patience=cfg["scheduler"].get("patience", 3),
                                      verbose=True)
        scheduler_type = "reduce_on_plateau"

    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    use_ft = cfg["loss"].get("use_focal_tversky", False)
    if use_ft:
        ft_cfg = cfg["loss"]["focal_tversky"]
        focal_tversky = FocalTverskyLoss(alpha=ft_cfg["alpha"], beta=ft_cfg["beta"], gamma=ft_cfg["gamma"])
    else:
        focal_tversky = None

    # AMP (use if CUDA available)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Logging + checkpointing
    tb_writer = None
    if cfg["logging"].get("use_tensorboard", True):
        log_dir = cfg["logging"].get("tb_log_dir", "runs") + f"/{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs -> {log_dir}")

    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    best_epoch = -1
    patience_counter = 0

    # Optionally resume
    resume_path = cfg["misc"].get("resume_from_checkpoint", "")
    start_epoch = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[INFO] Resumed from checkpoint {resume_path} at epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, cfg["training"]["epochs"]):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg['training']['epochs']}", leave=False)
            for batch_idx, (imgs, masks) in enumerate(loop):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(0)
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(imgs)
                    if masks.ndim == 3:
                        masks = masks.unsqueeze(1)

                    loss = cfg["loss"].get("bce_weight", 1.0) * bce_loss(outputs, masks)
                    if use_ft:
                        loss = loss + cfg["loss"].get("dice_weight", 1.0) * focal_tversky(outputs, masks)
                    else:
                        loss = loss + cfg["loss"].get("dice_weight", 1.0) * dice_loss(outputs, masks)

                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()

                if scheduler_type == "onecycle":
                    scheduler.step()

                train_loss += loss.item()
                loop.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / max(1, len(train_loader))

            # Validation
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc="Validation", leave=False):
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    if imgs.shape[1] == 1:
                        imgs = imgs.repeat(1, 3, 1, 1)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(imgs)
                        if masks.ndim == 3:
                            masks = masks.unsqueeze(1)
                        loss_val = cfg["loss"].get("bce_weight", 1.0) * bce_loss(outputs, masks)
                        if use_ft:
                            loss_val = loss_val + cfg["loss"].get("dice_weight", 1.0) * focal_tversky(outputs, masks)
                        else:
                            loss_val = loss_val + cfg["loss"].get("dice_weight", 1.0) * dice_loss(outputs, masks)

                    val_loss += loss_val.item()
                    val_dice += dice_coeff(outputs, masks)
                    val_iou += iou_score(outputs, masks)

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_dice = val_dice / max(1, len(val_loader))
            avg_val_iou = val_iou / max(1, len(val_loader))

            # Scheduler step for ReduceLROnPlateau
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(avg_val_loss)

            # Logging
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}")
            if tb_writer:
                tb_writer.add_scalar("Loss/train", avg_train_loss, epoch+1)
                tb_writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
                tb_writer.add_scalar("Metric/val_dice", avg_val_dice, epoch+1)
                tb_writer.add_scalar("Metric/val_iou", avg_val_iou, epoch+1)
                tb_writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch+1)

            # Checkpoint logic
            metric = avg_val_dice  # main metric to maximize
            is_best = False
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                patience_counter = 0
                is_best = True
                if cfg["training"].get("save_best_only", True):
                    save_path = ckpt_dir / f"best_epoch_{best_epoch}.pth"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                        "best_metric": best_metric
                    }, save_path)
                    print(f"[INFO] Saved best model to {save_path}")
            else:
                patience_counter += 1

            # periodic checkpoint
            if (epoch + 1) % cfg["training"].get("checkpoint_every_n_epochs", 5) == 0:
                cp_path = ckpt_dir / f"epoch_{epoch+1}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "val_loss": avg_val_loss,
                    "val_dice": avg_val_dice
                }, cp_path)
                print(f"[INFO] Periodic checkpoint saved to {cp_path}")

            # Early stopping
            if cfg["training"]["early_stopping"].get("enabled", True):
                if patience_counter >= cfg["training"]["early_stopping"].get("patience", 7):
                    print(f"[INFO] Early stopping triggered: no improvement for {patience_counter} epochs.")
                    break

    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user. Saving last checkpoint...")
        last_path = ckpt_dir / "interrupted_last.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, last_path)
        print(f"[INFO] Saved interrupted checkpoint to {last_path}")

    # Finalize
    print(f"[INFO] Training completed. Best val dice: {best_metric:.4f} at epoch {best_epoch}")
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    cfg_path = "config/config_segmentation.yml"
    train_from_config(cfg_path)
