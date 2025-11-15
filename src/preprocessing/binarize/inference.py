import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.preprocessing.binarize.model import UNetSmall
from src.preprocessing.binarize.dataset import FingerprintBinaryDataset


def load_config(path="config/config_binarize.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(weights_path, device):
    model = UNetSmall(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def preprocess_image(img_path, img_size=(256, 256)):
    """Legge un'immagine, la converte in tensor normalizzato e ridimensiona"""
    img = Image.open(img_path).convert("L")
    img = img.resize(img_size, resample=Image.BILINEAR)
    tensor = transforms.ToTensor()(img)
    tensor = tensor.unsqueeze(0)  # aggiunge batch dimension
    return tensor


def save_mask(mask_tensor, save_path):
    """Salva il tensore binario come PNG"""
    mask = mask_tensor.squeeze().cpu().numpy()  # [H,W]
    mask_img = Image.fromarray((mask * 255).astype("uint8"))
    mask_img.save(save_path)


def inference_on_folder(model, input_dir, output_dir, device, img_size=(256, 256), threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_segmented.png")])

    print(f"[INFO] {len(input_files)} immagini '_segmented.png' trovate in {input_dir}")
    
    for f in tqdm(input_files, desc="Inferencing"):
        img_path = os.path.join(input_dir, f)
        img_tensor = preprocess_image(img_path, img_size=img_size).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred_bin = (pred > threshold).float()

        save_path = os.path.join(output_dir, f)
        save_mask(pred_bin, save_path)


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("data/checkpoints/binarize/best_unet_small.pth", device)

    input_dir = cfg["paths"]["inference_dir"]       # cartella delle immagini da inferire
    output_dir = cfg["paths"]["inference_out_dir"]  # cartella dove salvare i risultati

    inference_on_folder(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        device=device,
        img_size=(256, 256),
        threshold=0.5
    )

    print("Inferenza completata!")


if __name__ == "__main__":
    main()
