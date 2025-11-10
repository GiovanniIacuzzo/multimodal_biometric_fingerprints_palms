import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, embedding_dim=512, freeze_backbone=True):
        super().__init__()
        self.model_name = model_name

        if model_name.startswith('resnet'):
            # Carica il modello ResNet
            self.backbone = getattr(models, model_name)(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            
            # Modifica il primo conv per input grayscale (1 canale)
            self.backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=self.backbone.conv1.bias is not None
            )
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name.startswith('efficientnet'):
            self.backbone = getattr(models, model_name)(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Proietta a embedding_dim se diverso da in_features
        if embedding_dim != in_features:
            self.projector = nn.Linear(in_features, embedding_dim)
            self.output_dim = embedding_dim
        else:
            self.projector = nn.Identity()
            self.output_dim = in_features

        # Freeze backbone se richiesto
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projector(features)
        return embedding
