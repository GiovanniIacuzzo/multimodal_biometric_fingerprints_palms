import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from src.preprocessing.thinning.model import UNet

# -------------------------------
# CONFIG
# -------------------------------
BINARY_DIR = "dataset/processed/debug/binary"   # input
SKELETON_DIR = "data/processed/debug/skeleton_inference"  # output
MODEL_PATH = "src/preprocessing/thinning/checkpoints/best_model.pth"  # modello salvato
IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SKELETON_DIR, exist_ok=True)

# -------------------------------
# Trasformazioni
# -------------------------------
transform_input = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# -------------------------------
# Load model
# -------------------------------
model = UNet(in_ch=1, out_ch=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------
# Inferenza
# -------------------------------
binary_files = sorted([f for f in Path(BINARY_DIR).iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

for bin_path in tqdm(binary_files, desc="Inferenza"):
    # Carica immagine
    img = Image.open(bin_path).convert("L")
    input_tensor = transform_input(img).unsqueeze(0).to(DEVICE)  # 1x1xHxW

    # Forward
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)  # assicura valori tra 0 e 1
        skeleton = (output > 0.5).float()  # binarizza
        skeleton = skeleton.squeeze().cpu().numpy() * 255  # 0-255
        skeleton_img = Image.fromarray(skeleton.astype(np.uint8))

    # Salva
    out_path = Path(SKELETON_DIR) / bin_path.name
    skeleton_img.save(out_path)
