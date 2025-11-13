import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

class ProjectionHead(nn.Module):
    """
    MLP per contrastive learning (SimCLR / BYOL-style).
    - Residual connection per stabilità
    - Dropout
    - Normalizzazione L2 sull’output
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        output_dim=128,
        num_layers=2,
        dropout=0.1,
        use_residual=True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers deve essere >= 1")

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(parametrizations.weight_norm(nn.Linear(input_dim, hidden_dim)))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers += [
                    parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.use_residual = use_residual and input_dim == output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        out = self.mlp(x)
        if self.use_residual:
            out = out + x
        out = F.normalize(out, dim=1, p=2)
        return out
