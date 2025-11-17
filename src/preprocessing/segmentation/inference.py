import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
import logging
from colorama import Fore, Style

from src.preprocessing.segmentation.model import FingerprintSegmentationModel

# ====================================================
# LOGGING SETUP
# ====================================================
OUTPUT_DIR = "data/metadata"
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "inference.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def console_step(title: str):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

# ====================================================
# CONFIG
# ====================================================
with open("config/config_segmentation.yml", "r") as f:
    cfg = yaml.safe_load(f)

IMG_DIR = cfg["dataset"]["img_dir"]
OUTPUT_DIR = cfg["dataset"]["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = cfg["training"].get(
    "best_checkpoint_path",
    "data/checkpoints/segmentation/best_epoch_1.pth"
)
IMAGE_SIZE = tuple(cfg["dataset"]["image_size"])

# ====================================================
# DEVICE
# ====================================================
dev_choice = cfg["training"]["device"].lower()
if dev_choice == "auto":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(dev_choice)

console_step("Device Setup")
print(f"{Fore.GREEN}Using device:{Style.RESET_ALL} {DEVICE}")
logging.info(f"Using device: {DEVICE}")

# ====================================================
# MODEL
# ====================================================
console_step("Loading Model")
model = FingerprintSegmentationModel(
    num_labels=cfg["model"]["num_labels"],
    image_size=IMAGE_SIZE,
    pretrained_model=cfg["model"]["pretrained_model"]
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"{Fore.GREEN}✔ Model loaded from:{Style.RESET_ALL} {MODEL_PATH}")
logging.info(f"Model loaded from {MODEL_PATH}")

# ====================================================
# FUNCTIONS
# ====================================================
def preprocess_image(img_path, image_size):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.imread(img_path)
    img_resized = cv2.resize(img_gray, image_size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)  # replicate channels
    return img_tensor, img_rgb

def mask_to_rgb(mask_tensor, original_rgb, threshold=0.5):
    mask = torch.sigmoid(mask_tensor).squeeze().cpu().detach().numpy()
    mask_bin = (mask > threshold).astype(np.uint8)
    mask_bin_resized = cv2.resize(mask_bin, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = original_rgb.copy()
    overlay[mask_bin_resized == 0] = 0

    # Colored overlay (red)
    mask_color = np.zeros_like(original_rgb)
    mask_color[:, :, 2] = mask_bin_resized * 255
    overlay_color = cv2.addWeighted(original_rgb, 0.7, mask_color, 0.3, 0)

    return mask_bin_resized * 255, overlay, overlay_color

# ====================================================
# INFERENZA
# ====================================================
console_step("Inferenza su Dataset")
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for fname in tqdm(image_files, desc="Inferenza", ncols=90):
    img_path = os.path.join(IMG_DIR, fname)
    img_tensor, original_rgb = preprocess_image(img_path, IMAGE_SIZE)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    mask_bin, segmented, overlay_color = mask_to_rgb(output, original_rgb)

    fname_base = Path(fname).stem
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname_base}_mask.png"), mask_bin)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname_base}_segmented.png"), segmented)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname_base}_overlay.png"), overlay_color)

    logging.info(f"Processed: {fname}")
    print(f"{Fore.GREEN}✔{Style.RESET_ALL} {fname} processata")

console_step("Inferenza Completata")
logging.info(f"Inferenza completata. Output salvati in {OUTPUT_DIR}")
print(f"{Fore.CYAN}✨ Inferenza completata. Output salvati in '{OUTPUT_DIR}' ✨{Style.RESET_ALL}")
