# src/preprocessing/segmentation/inference.py
import os
import torch
import cv2
from tqdm import tqdm
from src.preprocessing.segmentation.model import FingerprintSegmentationModel
import numpy as np

# --- Configurazioni ---
IMG_DIR = "dataset/DBII"
OUTPUT_DIR = "data/processed/segmentation"
MODEL_PATH = "src/preprocessing/segmentation/model/segmentation.pth"
IMAGE_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Carica modello ---
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)

# --- Preprocessing ---
def preprocess_image(img_path, image_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.tensor(img/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)  # 3 canali
    return img_tensor, cv2.imread(img_path)  # ritorna anche RGB originale

# --- Postprocessing ---
def mask_to_rgb(mask_tensor, original_rgb):
    # Sigmoid per normalizzare in [0,1]
    mask = torch.sigmoid(mask_tensor).squeeze().cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8)  # binarizza
    mask_rgb = cv2.resize(mask, (original_rgb.shape[1], original_rgb.shape[0]))
    
    # Applica la maschera all'immagine originale
    segmented = original_rgb.copy()
    segmented[mask_rgb == 0] = 0  # nero dove non c'è impronta
    return segmented

# --- Inferenza ---
for fname in tqdm(os.listdir(IMG_DIR)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    img_tensor, original_rgb = preprocess_image(img_path, IMAGE_SIZE)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        segmented_rgb = mask_to_rgb(output, original_rgb)

    output_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(output_path, segmented_rgb)

print(f"Inferenza completata. Maschere RGB salvate in '{OUTPUT_DIR}'")
