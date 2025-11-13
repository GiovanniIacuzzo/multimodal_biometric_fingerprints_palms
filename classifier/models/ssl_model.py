from classifier.models.backbone import FingerprintViTBackbone
from classifier.models.projection_head import ProjectionHead
import torch.nn as nn
import torch

class SSLModel(nn.Module):
    def __init__(self,
                 backbone_name="vit_base_patch16_224",
                 pretrained=True,
                 checkpoint_path=None,
                 embedding_dim=256,
                 proj_hidden_dim=512,
                 proj_output_dim=128,
                 proj_num_layers=2,
                 freeze_backbone=False,
                 use_predictor=True):
        super().__init__()
        self.backbone = FingerprintViTBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone,
            norm_layer=True,
            dropout=0.2
        )
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            num_layers=proj_num_layers
        )
        self.predictor = ( nn.Sequential(
                                nn.Linear(proj_output_dim, proj_hidden_dim),
                                nn.BatchNorm1d(proj_hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(proj_hidden_dim, proj_output_dim)
                           ) if use_predictor else nn.Identity() )
        self.use_predictor = use_predictor

    def forward(self, x: torch.Tensor, return_embedding: bool=False):
        embedding = self.backbone(x)
        projection = self.projection_head(embedding)
        projection_pred = self.predictor(projection) if self.use_predictor else projection
        if return_embedding:
            return projection_pred, embedding
        return projection_pred
