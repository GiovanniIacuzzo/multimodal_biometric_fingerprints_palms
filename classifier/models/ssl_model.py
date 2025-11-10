import torch
import torch.nn as nn
from classifier.models.backbone import CNNBackbone
from classifier.models.projection_head import ProjectionHead

class SSLModel(nn.Module):
    """
    Wrapper per Self-Supervised Learning:
    backbone CNN + projection head.
    """

    def __init__(self, 
                 backbone_name='resnet18', 
                 pretrained=True, 
                 embedding_dim=256, 
                 proj_hidden_dim=512, 
                 proj_output_dim=128, 
                 proj_num_layers=2, 
                 freeze_backbone=False):
        super().__init__()

        # Backbone CNN
        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone
        )

        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            num_layers=proj_num_layers
        )

    def forward(self, x, return_embedding=False):
        """
        x: batch di immagini (B, C, H, W)
        return_embedding: se True restituisce anche embedding backbone (prima della proiezione)
        """
        if not isinstance(return_embedding, bool):
            raise ValueError(f"return_embedding deve essere un bool, ricevuto {type(return_embedding)}")

        embedding = self.backbone(x)          # embedding dal backbone
        z = self.projection_head(embedding)   # embedding proiettato per contrastive loss

        if return_embedding:
            return z, embedding
        return z
