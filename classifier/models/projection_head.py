import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    MLP projection head per contrastive learning (SimCLR-style).
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=128, num_layers=2):
        """
        input_dim: dimensione embedding dal backbone
        hidden_dim: dimensione layer nascosto
        output_dim: dimensione finale dello spazio latente
        num_layers: numero di layer MLP
        """
        super().__init__()

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: embedding dal backbone
        """
        z = self.mlp(x)
        # Normalizzazione L2
        z = F.normalize(z, dim=1)
        return z
