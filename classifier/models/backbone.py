import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(nn.Module):
    """
    Backbone CNN modulare per feature extraction da immagini biometriche (es. impronte digitali).
    Supporta ResNet ed EfficientNet, con conversione automatica per input grayscale.
    """

    SUPPORTED_MODELS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
    }

    def __init__(self, model_name="resnet18", pretrained=True, embedding_dim=512, freeze_backbone=True):
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(self.SUPPORTED_MODELS.keys())}")

        # Caricamento dinamico modello
        self.model_name = model_name
        backbone_fn = self.SUPPORTED_MODELS[model_name]

        # Carica pesi pretrained corretti per modello
        try:
            self.backbone = backbone_fn(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            self.backbone = backbone_fn(weights=None)

        # Adatta primo conv per input grayscale (1 canale)
        first_conv = list(self.backbone.children())[0]
        if isinstance(first_conv, nn.Conv2d) and first_conv.in_channels != 1:
            self.backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )

        # Rimuove classificatore finale
        if "resnet" in model_name:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "efficientnet" in model_name:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unexpected model type: {model_name}")

        # Proiezione opzionale
        self.projector = nn.Linear(in_features, embedding_dim) if embedding_dim != in_features else nn.Identity()
        self.output_dim = embedding_dim

        # Freeze backbone se richiesto
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        # Controllo batch dimensione
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape (B, C, H, W), got {x.shape}")

        features = self.backbone(x)
        embedding = self.projector(features)
        return embedding
