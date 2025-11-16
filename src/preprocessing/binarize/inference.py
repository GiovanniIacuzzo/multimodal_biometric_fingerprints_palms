import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from src.preprocessing.binarize.model import UNet

# ---------------------------
# Config loader
# ---------------------------
def load_config(path="config/config_binarize.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------
# Model loader
# ---------------------------
def load_model(weights_path, device, base_ch):
    model = UNet(in_ch=1, out_ch=1, base_ch=base_ch)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    return model

# ---------------------------
# Preprocess singola immagine
# ---------------------------
def preprocess_image(img_path, img_size=(256,256)):
    img = Image.open(img_path).convert("L")
    img = img.resize(img_size, resample=Image.BILINEAR)
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1,1,H,W]
    return tensor

# ---------------------------
# Salvataggio mask
# ---------------------------
def save_mask(mask_tensor, save_path):
    mask = mask_tensor.squeeze().cpu().numpy()  # [H,W]
    mask_img = Image.fromarray((mask*255).astype("uint8"))
    mask_img.save(save_path)

# ---------------------------
# Inferenza su cartella
# ---------------------------
def inference_on_folder(model, input_dir, output_dir, device, img_size=(256,256), threshold=0.5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
    if len(input_files) == 0:
        print(f"[WARNING] Nessuna immagine trovata in {input_dir}")
        return

    for f in tqdm(input_files, desc="Inferencing"):
        img_path = input_dir / f
        img_tensor = preprocess_image(img_path, img_size=img_size).to(device)

        with torch.no_grad():
            pred = model(img_tensor)           # [1,1,H,W]
            pred = torch.sigmoid(pred)
            pred_bin = (pred > threshold).float()

        save_path = output_dir / f
        save_mask(pred_bin, save_path)

# ---------------------------
# Main
# ---------------------------
def main(config_path="config/config_binarize.yml"):
    cfg = load_config(config_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")

    # Checkpoint U-Net
    weights_path = Path(cfg["training"].get("ckpt_dir", "data/checkpoints/binarize")) / "last_unet.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {weights_path}")

    model = load_model(weights_path=weights_path, device=device, base_ch=cfg["model"].get("base_channels",64))

    input_dir = cfg["paths"]["inference_dir"]
    output_dir = cfg["paths"]["inference_out_dir"]
    img_size = tuple(cfg["dataset"].get("img_size",(256,256)))
    threshold = cfg["training"].get("threshold",0.5)

    inference_on_folder(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        device=device,
        img_size=img_size,
        threshold=threshold
    )

    print("[INFO] Inferenza completata!")

# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_binarize.yml")
    args = parser.parse_args()
    main(args.config)
