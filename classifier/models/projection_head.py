import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Multi-layer projection head per contrastive learning (SimCLR-style).
    Include normalizzazione L2 e dropout per stabilità numerica.
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers deve essere >= 1")

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ]
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        z = self.mlp(x)
        z = F.normalize(z, dim=1, p=2)
        return z
