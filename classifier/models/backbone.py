import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    """
    Backbone CNN per feature extraction da immagini biometriche (es. impronte).
    Supporta ResNet ed EfficientNet, con:
      - adattamento automatico a input grayscale
      - pooling adattivo
      - estrazione multiscala opzionale
    """

    SUPPORTED_MODELS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
    }

    def __init__(
        self,
        model_name="resnet18",
        pretrained=True,
        embedding_dim=512,
        freeze_backbone=False,
        use_multiscale=True,
        norm_layer=True,
        dropout=0.2,
    ):
        super().__init__()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(self.SUPPORTED_MODELS.keys())}")

        self.model_name = model_name  # FIX: serve per forward
        self.use_multiscale = use_multiscale
        self.output_dim = embedding_dim

        backbone_fn = self.SUPPORTED_MODELS[model_name]
        try:
            self.backbone = backbone_fn(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            self.backbone = backbone_fn(weights=None)

        # Adatta primo conv a grayscale con kernel più piccolo
        first_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=3,  # da 7x7 a 3x3
            stride=1,
            padding=1,
            bias=False,
        )

        # Rimuove classificatore
        if "resnet" in model_name:
            self.feature_layers = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            )
        elif "efficientnet" in model_name:
            self.feature_layers = self.backbone.features
        else:
            raise ValueError(f"Unexpected model type: {model_name}")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LayerNorm + Dropout opzionali
        self.norm = nn.LayerNorm(embedding_dim) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze backbone se richiesto
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Il projector lo inizializziamo dinamicamente nel primo forward
        self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape (B, C, H, W), got {x.shape}")

        if "resnet" in self.model_name:
            feats = []
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            for layer in [self.backbone.layer1, self.backbone.layer2,
                          self.backbone.layer3, self.backbone.layer4]:
                x = layer(x)
                if self.use_multiscale:
                    pooled = self.global_pool(x)
                    feats.append(pooled.flatten(1))

            feat = torch.cat(feats, dim=1) if self.use_multiscale else self.global_pool(x).flatten(1)
        else:
            feat = self.global_pool(self.feature_layers(x)).flatten(1)

        # Inizializza projector dinamicamente al primo forward
        if self.projector is None:
            self.projector = nn.Linear(feat.shape[1], self.output_dim).to(x.device)

        embedding = self.projector(feat)
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        return embedding
