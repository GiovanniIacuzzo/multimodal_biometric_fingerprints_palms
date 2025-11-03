from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================
# FEATURE EXTRACTOR
# ================================================
class FingerprintFeatureExtractor(nn.Module):
    """
    Estrae feature L2-normalizzate da un backbone (ResNet, EfficientNet, ViT...).
    """
    def __init__(self, embedding_dim=256, backbone="resnet50", pretrained=True):
        super().__init__()

        # Selezione backbone
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Backbone non supportato: {backbone}")

        self.features = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.fc = nn.Linear(in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)              # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, dim=1)         # normalizzazione L2
        return x