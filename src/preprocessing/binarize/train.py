import yaml
import time
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.preprocessing.binarize.dataset import FingerprintBinaryDataset
from src.preprocessing.binarize.model import UNet

def load_config(path="config/config_binarize.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()

def threshold_preds(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).float()

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> dict:
    preds = threshold_preds(logits, thr)
    targets = targets.float()
    eps = 1e-6

    tp = (preds * targets).sum(dim=(1,2,3))
    fp = (preds * (1 - targets)).sum(dim=(1,2,3))
    fn = ((1 - preds) * targets).sum(dim=(1,2,3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(1,2,3))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    return {
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "acc": acc.mean().item()
    }

def save_batch_visuals(img_in: torch.Tensor, logits: torch.Tensor, gt: torch.Tensor,
                       out_dir: Path, epoch:int, prefix: str = "vis", n: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(logits)
    n = min(n, img_in.size(0))
    for i in range(n):
        save_image(img_in[i], out_dir / f"{prefix}_epoch{epoch:03d}_idx{i:02d}_inp.png")
        save_image(probs[i], out_dir / f"{prefix}_epoch{epoch:03d}_idx{i:02d}_prob.png")
        save_image((probs[i]>0.5).float(), out_dir / f"{prefix}_epoch{epoch:03d}_idx{i:02d}_pred.png")
        save_image(gt[i], out_dir / f"{prefix}_epoch{epoch:03d}_idx{i:02d}_gt.png")

def train_one_epoch(model, loader, optimizer, criterion_bce, criterion_dice,
                    device, scaler, epoch, cfg, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    running_metrics = {"f1":0.0,"iou":0.0,"dice":0.0,"acc":0.0}
    steps = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for img_in, img_gt in pbar:
        img_in = img_in.to(device)
        img_gt = img_gt.to(device)
        img_gt_bin = (img_gt > 0.5).float()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(img_in)
            loss = criterion_bce(logits, img_gt_bin) + cfg["training"].get("dice_weight",1.0)*criterion_dice(logits, img_gt_bin)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm>0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_metrics = compute_metrics(logits.detach(), img_gt_bin.detach(),
                                        thr=cfg["training"].get("threshold",0.5))
        for k in running_metrics: running_metrics[k] += batch_metrics[k]
        running_loss += loss.item()
        steps += 1
        pbar.set_postfix({"loss": f"{running_loss/steps:.4f}", "dice": f"{running_metrics['dice']/steps:.4f}"})
    for k in running_metrics: running_metrics[k] /= max(1, steps)
    return running_loss/max(1,steps), running_metrics

def validate(model, loader, criterion_bce, criterion_dice, device, cfg, epoch, vis_dir: Path=None):
    model.eval()
    running_loss = 0.0
    running_metrics = {"f1":0.0,"iou":0.0,"dice":0.0,"acc":0.0}
    steps = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val   E{epoch}", leave=False)
        for img_in, img_gt in pbar:
            img_in = img_in.to(device)
            img_gt = img_gt.to(device)
            img_gt_bin = (img_gt>0.5).float()

            logits = model(img_in)
            loss = criterion_bce(logits,img_gt_bin) + cfg["training"].get("dice_weight",1.0)*criterion_dice(logits,img_gt_bin)
            batch_metrics = compute_metrics(logits, img_gt_bin, thr=cfg["training"].get("threshold",0.5))
            for k in running_metrics: running_metrics[k]+=batch_metrics[k]
            running_loss += loss.item()
            steps+=1

            pbar.set_postfix({"loss": f"{running_loss/steps:.4f}", "dice": f"{running_metrics['dice']/steps:.4f}"})
            if vis_dir is not None and steps==1:
                save_batch_visuals(img_in.cpu(), logits.cpu(), img_gt_bin.cpu(), vis_dir, epoch, prefix="val")

    for k in running_metrics: running_metrics[k]/=max(1,steps)
    return running_loss/max(1,steps), running_metrics

def create_model(cfg, device):
    base_ch = cfg["model"].get("base_channels",64)
    model = UNet(in_ch=1,out_ch=1,base_ch=base_ch).to(device)
    checkpoint_path = cfg["training"].get("checkpoint_path",None)
    if checkpoint_path is not None:
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path,map_location=device)
        model.load_state_dict(state_dict,strict=False)
    return model

def main(config_path="config/config_binarize.yml"):
    cfg = load_config(config_path)

    # Seeds
    seed = cfg["training"].get("seed",42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Directories
    ckpt_dir = Path(cfg["training"].get("ckpt_dir","data/checkpoints/binarize"))
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    vis_dir = Path(cfg["training"].get("vis_dir","data/vis/binarize"))
    vis_dir.mkdir(parents=True,exist_ok=True)

    # Dataset + loaders
    train_ds = FingerprintBinaryDataset(cfg["paths"]["train_dir"], split="train",
                                       val_split=cfg["paths"]["val_split"],
                                       subset_size=cfg["dataset"].get("subset_size",-1),
                                       subset_seed=cfg["dataset"].get("subset_seed",42),
                                       img_size=tuple(cfg["dataset"].get("img_size",(256,256))),
                                       augment=cfg["dataset"].get("augment",True))
    val_ds = FingerprintBinaryDataset(cfg["paths"]["train_dir"], split="val",
                                     val_split=cfg["paths"]["val_split"],
                                     subset_size=cfg["dataset"].get("subset_size",-1),
                                     subset_seed=cfg["dataset"].get("subset_seed",42),
                                     img_size=tuple(cfg["dataset"].get("img_size",(256,256))),
                                     augment=False)
    train_loader = DataLoader(train_ds,batch_size=cfg["training"].get("batch_size",8),
                              shuffle=True,num_workers=cfg["training"].get("num_workers",4),pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=cfg["training"].get("batch_size",8),
                            shuffle=False,num_workers=cfg["training"].get("num_workers",4),pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = create_model(cfg, device)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"].get("lr",1e-3),
                                  weight_decay=cfg["training"].get("weight_decay",1e-5))

    scheduler_cfg = cfg["training"].get("scheduler",{"type":"ReduceLROnPlateau","patience":5})
    if scheduler_cfg.get("type","ReduceLROnPlateau")=="CosineWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=scheduler_cfg.get("T_0",10), T_mult=scheduler_cfg.get("T_mult",1)
        )
        scheduler_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_cfg.get("factor",0.5),
            patience=scheduler_cfg.get("patience",5), min_lr=scheduler_cfg.get("min_lr",1e-7)
        )
        scheduler_plateau = True

    use_amp = cfg["training"].get("amp",True) and device.type=="cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs = cfg["training"].get("epochs",50)
    save_every = cfg["training"].get("save_every",10)
    patience = cfg["training"].get("early_stopping_patience",None)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(model,train_loader,optimizer,
                                                    criterion_bce,criterion_dice,
                                                    device,scaler,epoch,cfg,
                                                    max_grad_norm=cfg["training"].get("max_grad_norm",1.0))
        val_loss, val_metrics = validate(model,val_loader,criterion_bce,criterion_dice,
                                         device,cfg,epoch,vis_dir if cfg["training"].get("save_val_visuals",True) else None)

        if scheduler_plateau:
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch + val_loss/(train_loss+1e-8))

        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_dice={train_metrics['dice']:.4f} val_dice={val_metrics['dice']:.4f} "
              f"time={(time.time()-t0):.1f}s")

        # Checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_unet_fingerprint.pth")
            print("✓ Saved best model (val loss improved)")
        else:
            epochs_no_improve += 1

        if epoch % save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}_unet.pth")
        torch.save(model.state_dict(), ckpt_dir / "last_unet.pth")

        if (patience is not None) and (epochs_no_improve >= patience):
            print(f"[EarlyStopping] No improvement for {epochs_no_improve} epochs. Stopping.")
            break

    print("Training finished.")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="config/config_binarize.yml")
    args = parser.parse_args()
    main(args.config)
