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
from colorama import Fore, Style, init

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except Exception:
    HAS_ALB = False

from torch.utils.tensorboard import SummaryWriter
from src.preprocessing.segmentation.dataset import FingerprintDataset
from src.preprocessing.segmentation.model import FingerprintSegmentationModel

init(autoreset=True)

# ====================================================
# Utility functions
# ====================================================
def console_step(title):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================
# Losses & Metrics
# ====================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = targets.float()
        TP = (preds * targets).sum(dim=(1,2,3))
        FP = (preds * (1 - targets)).sum(dim=(1,2,3))
        FN = ((1 - preds) * targets).sum(dim=(1,2,3))
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return ((1 - tversky) ** self.gamma).mean()


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - ((2 * inter + eps)/(union + eps)).mean()


def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = (target > 0.5)
    inter = (pred & target).sum(dim=(1,2,3)).float()
    union = pred.sum(dim=(1,2,3)).float() + target.sum(dim=(1,2,3)).float()
    return ((2*inter + eps)/(union + eps)).mean().item()


def iou_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = (target > 0.5)
    inter = (pred & target).sum(dim=(1,2,3)).float()
    union = (pred | target).sum(dim=(1,2,3)).float()
    return ((inter + eps)/(union + eps)).mean().item()


# ====================================================
# Transforms
# ====================================================
def build_transforms(cfg):
    img_h, img_w = cfg["dataset"]["image_size"]
    if cfg["augmentation"]["use_albumentations"] and HAS_ALB:
        aug = [A.Resize(img_h, img_w)]
        if cfg["augmentation"]["horizontal_flip_p"] > 0:
            aug.append(A.HorizontalFlip(p=cfg["augmentation"]["horizontal_flip_p"]))
        ssr = cfg["augmentation"]["shift_scale_rotate"]
        aug.append(A.ShiftScaleRotate(
            shift_limit=ssr["shift_limit"],
            scale_limit=ssr["scale_limit"],
            rotate_limit=ssr["rotate_limit"],
            p=ssr["p"]
        ))
        if cfg["augmentation"]["brightness_contrast_p"] > 0:
            aug.append(A.RandomBrightnessContrast(p=cfg["augmentation"]["brightness_contrast_p"]))
        if cfg["augmentation"]["gauss_noise_p"] > 0:
            aug.append(A.GaussNoise(p=cfg["augmentation"]["gauss_noise_p"]))
        if cfg["augmentation"]["elastic_transform_p"] > 0:
            aug.append(A.ElasticTransform(p=cfg["augmentation"]["elastic_transform_p"]))
        aug.append(ToTensorV2())
        return A.Compose(aug)
    else:
        # fallback simple transform
        def simple_transform(img, mask):
            from PIL import Image
            import torchvision.transforms as T
            pil_img = Image.fromarray(img)
            pil_mask = Image.fromarray(mask)
            pil_img = pil_img.resize((img_w, img_h))
            pil_mask = pil_mask.resize((img_w, img_h))
            return T.ToTensor()(pil_img), T.ToTensor()(pil_mask)
        return simple_transform

def collect_image_mask_paths(img_dir, mask_base_dir):
    # Tutti i file immagine
    img_paths = sorted(Path(img_dir).rglob("*.jpg"))
    
    # Tutti i file mask
    mask_paths = sorted(Path(mask_base_dir).rglob("*/mask/*.jpg"))
    
    # Verifica che corrispondano in nome (se necessario)
    img_dict = {p.stem: p for p in img_paths}
    mask_dict = {p.stem: p for p in mask_paths}
    
    common_keys = set(img_dict.keys()) & set(mask_dict.keys())
    
    imgs_final = [img_dict[k] for k in common_keys]
    masks_final = [mask_dict[k] for k in common_keys]
    
    return imgs_final, masks_final

def collate_fn(batch):
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)


# ====================================================
# Main Training Function
# ====================================================
def train_from_config(cfg_path: str):
    # --- Load config ---
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    console_step("Setup")
    set_seed(cfg["experiment"].get("seed", 42))

    # Device
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
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Using device: {device}")

    # Transforms
    transform = build_transforms(cfg)

    # Instantiate dataset
    img_paths, mask_paths = collect_image_mask_paths(
        cfg["dataset"]["img_dir"], 
        cfg["dataset"]["mask_dir_base"]
    )

    dataset = FingerprintDataset(
        img_paths,
        mask_paths,
        image_size=tuple(cfg["dataset"]["image_size"]),
        transform=transform
    )
    N = len(dataset)
    val_split = cfg["dataset"].get("val_split", 0.2)
    indices = np.arange(N)
    np.random.shuffle(indices)
    val_count = int(N*val_split)
    val_idx, train_idx = indices[:val_count], indices[val_count:]

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=cfg["dataset"]["batch_size"],
                              shuffle=cfg["dataset"]["shuffle"],
                              num_workers=cfg["dataset"]["num_workers"],
                              collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=cfg["dataset"]["batch_size"],
                            shuffle=False,
                            num_workers=max(0, cfg["dataset"]["num_workers"]//2),
                            collate_fn=collate_fn)

    # Model
    model = FingerprintSegmentationModel(cfg["model"]["num_labels"],
                                         tuple(cfg["dataset"]["image_size"]),
                                         pretrained_model=cfg["model"]["pretrained_model"]).to(device)

    optimizer = AdamW(model.parameters(),
                      lr=cfg["optimizer"]["lr"],
                      weight_decay=cfg["optimizer"]["weight_decay"])

    # Scheduler
    if cfg["scheduler"]["type"] == "OneCycleLR":
        scheduler = OneCycleLR(optimizer,
                               max_lr=cfg["optimizer"]["lr"],
                               steps_per_epoch=len(train_loader),
                               epochs=cfg["training"]["epochs"])
        scheduler_type = "onecycle"
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                      factor=cfg["scheduler"].get("factor", 0.5),
                                      patience=cfg["scheduler"].get("patience", 3),
                                      verbose=True)
        scheduler_type = "reduce_on_plateau"

    # Loss
    bce_loss = nn.BCEWithLogitsLoss()
    use_ft = cfg["loss"].get("use_focal_tversky", False)
    focal_tversky = FocalTverskyLoss(**cfg["loss"]["focal_tversky"]) if use_ft else None

    # AMP
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # TensorBoard
    tb_writer = None
    if cfg["logging"].get("use_tensorboard", True):
        log_dir = Path(cfg["logging"].get("tb_log_dir", "runs")) / f"{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir)
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} TensorBoard logs -> {log_dir}")

    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_metric, best_epoch, patience_counter = -1.0, -1, 0

    # Resume checkpoint
    start_epoch = 0
    resume_path = cfg["misc"].get("resume_from_checkpoint", "")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Resumed from checkpoint {resume_path} at epoch {start_epoch}")

    console_step("Training Loop")
    try:
        for epoch in range(start_epoch, cfg["training"]["epochs"]):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"{Fore.CYAN}Train Epoch {epoch+1}/{cfg['training']['epochs']}{Style.RESET_ALL}", leave=False)
            for imgs, masks in loop:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(imgs)
                    loss = bce_loss(outputs, masks)
                    if use_ft: loss += focal_tversky(outputs, masks)
                    else: loss += dice_loss(outputs, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm",1.0))
                scaler.step(optimizer)
                scaler.update()
                if scheduler_type=="onecycle": scheduler.step()
                train_loss += loss.item()
                loop.set_postfix({"loss": f"{loss.item():.4f}"})
            avg_train_loss = train_loss / max(1, len(train_loader))

            # Validation
            model.eval()
            val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc=f"{Fore.MAGENTA}Validation{Style.RESET_ALL}", leave=False):
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    loss_val = bce_loss(outputs, masks)
                    if use_ft: loss_val += focal_tversky(outputs, masks)
                    else: loss_val += dice_loss(outputs, masks)
                    val_loss += loss_val.item()
                    val_dice += dice_coeff(outputs, masks)
                    val_iou += iou_score(outputs, masks)
            avg_val_loss = val_loss / max(1,len(val_loader))
            avg_val_dice = val_dice / max(1,len(val_loader))
            avg_val_iou = val_iou / max(1,len(val_loader))

            if scheduler_type=="reduce_on_plateau": scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}")
            if tb_writer:
                tb_writer.add_scalar("Loss/train", avg_train_loss, epoch+1)
                tb_writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
                tb_writer.add_scalar("Metric/val_dice", avg_val_dice, epoch+1)
                tb_writer.add_scalar("Metric/val_iou", avg_val_iou, epoch+1)

            # Checkpoint
            metric = avg_val_dice
            if metric > best_metric:
                best_metric, best_epoch, patience_counter = metric, epoch+1, 0
                save_path = ckpt_dir / f"best_epoch_{best_epoch}.pth"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_metric": best_metric, "config": cfg}, save_path)
                print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Saved best model to {save_path}")
            else:
                patience_counter += 1

            # Periodic checkpoint
            if (epoch+1) % cfg["training"].get("checkpoint_every_n_epochs", 5) == 0:
                cp_path = ckpt_dir / f"epoch_{epoch+1}.pth"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": avg_val_loss, "val_dice": avg_val_dice}, cp_path)
                print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Periodic checkpoint saved to {cp_path}")

            # Early stopping
            if cfg["training"]["early_stopping"].get("enabled", True) and patience_counter >= cfg["training"]["early_stopping"].get("patience", 7):
                print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Early stopping triggered at epoch {epoch+1}")
                break

    except KeyboardInterrupt:
        print(f"{Fore.RED}[INFO]{Style.RESET_ALL} Training interrupted. Saving last checkpoint...")
        last_path = ckpt_dir / "interrupted_last.pth"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, last_path)
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Saved interrupted checkpoint to {last_path}")

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Training completed. Best val dice: {best_metric:.4f} at epoch {best_epoch}")
    if tb_writer: tb_writer.close()


if __name__ == "__main__":
    cfg_path = "config/config_segmentation.yml"
    train_from_config(cfg_path)
