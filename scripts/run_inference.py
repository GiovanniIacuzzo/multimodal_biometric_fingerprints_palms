import os
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.FeatureExtractor import FingerprintFeatureExtractor

# Percorso del modello addestrato
MODEL_PATH = "data/model/deep_descriptor.pth"

# Inizializza il modello
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🖥 Device inferenza: {device}")

model = FingerprintFeatureExtractor(
    embedding_dim=256,  # deve corrispondere a quello del training
    backbone="resnet50",
    pretrained=False
).to(device)

# Carica pesi addestrati
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Trasformazione come nel training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Percorso immagine da testare
img_path = "dataset/DBII/1_1_1.jpg"
img = Image.open(img_path).convert("L")
x = transform(img).unsqueeze(0).to(device)

# Estrazione feature
with torch.no_grad():
    embedding = model(x).cpu().numpy().flatten()

print(f"Feature estratte da {os.path.basename(img_path)}")
print("Dimensione embedding:", embedding.shape)
print("Esempio valori:", embedding[:10])
