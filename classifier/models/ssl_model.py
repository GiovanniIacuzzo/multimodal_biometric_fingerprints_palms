import torch
import torch.nn as nn
from classifier.models.backbone import CNNBackbone
from classifier.models.projection_head import ProjectionHead

class SSLModel(nn.Module):
    """
    Self-Supervised Model: backbone CNN + projection head.
    Gestisce sia l'output proiettato per la contrastive loss
    sia l'embedding grezzo per analisi o clustering.
    """

    def __init__(self,
                 backbone_name="resnet18",
                 pretrained=True,
                 embedding_dim=256,
                 proj_hidden_dim=512,
                 proj_output_dim=128,
                 proj_num_layers=2,
                 freeze_backbone=False):
        super().__init__()

        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone
        )

        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            num_layers=proj_num_layers
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

        embedding = self.backbone(x)
        projection = self.projection_head(embedding)

        return (projection, embedding) if return_embedding else projection
