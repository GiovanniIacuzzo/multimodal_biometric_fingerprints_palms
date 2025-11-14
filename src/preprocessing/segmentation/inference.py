import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
from src.preprocessing.segmentation.model import FingerprintSegmentationModel

# ----------------------------
# Config
# ----------------------------
with open("config/config_segmentation.yml", "r") as f:
    cfg = yaml.safe_load(f)

IMG_DIR = cfg["dataset"]["img_dir"]
OUTPUT_DIR = cfg["dataset"]["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = cfg["training"].get("best_checkpoint_path", "data/checkpoints/segmentation/best_epoch_n.pth")
IMAGE_SIZE = tuple(cfg["dataset"]["image_size"])

# ----------------------------
# Device
# ----------------------------
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

print(f"[INFO] Using device: {DEVICE}")

# ----------------------------
# Carica modello
# ----------------------------
model = FingerprintSegmentationModel(
    num_labels=cfg["model"]["num_labels"],
    image_size=IMAGE_SIZE,
    pretrained_model=cfg["model"]["pretrained_model"]
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print(f"[INFO] Loaded model from {MODEL_PATH}")

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(img_path, image_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_rgb = cv2.imread(img_path)
    img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)  # replicate channels
    return img_tensor, original_rgb

# ----------------------------
# Postprocessing
# ----------------------------
def mask_to_rgb(mask_tensor, original_rgb, threshold=0.5):
    # Sigmoid -> binarizza
    mask = torch.sigmoid(mask_tensor).squeeze().cpu().detach().numpy()
    mask_bin = (mask > threshold).astype(np.uint8)  # maschera binaria
    mask_bin_resized = cv2.resize(mask_bin, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay semitrasparente
    overlay = original_rgb.copy()
    overlay[mask_bin_resized == 0] = 0

    # Optional: maschera colorata (rosso) sovrapposta
    mask_color = np.zeros_like(original_rgb)
    mask_color[:, :, 2] = mask_bin_resized * 255  # rosso
    overlay_color = cv2.addWeighted(original_rgb, 0.7, mask_color, 0.3, 0)

    return mask_bin_resized * 255, overlay, overlay_color

# ----------------------------
# Inferenza
# ----------------------------
for fname in tqdm(os.listdir(IMG_DIR), desc="Inferenza"):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    img_tensor, original_rgb = preprocess_image(img_path, IMAGE_SIZE)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    mask_bin, segmented, overlay_color = mask_to_rgb(output, original_rgb)

    # Salva output
    fname_base = Path(fname).stem
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname_base}_mask.png"), mask_bin)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname_base}_segmented.png"), segmented)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname_base}_overlay.png"), overlay_color)

print(f"[INFO] Inferenza completata. Output salvati in '{OUTPUT_DIR}'")
