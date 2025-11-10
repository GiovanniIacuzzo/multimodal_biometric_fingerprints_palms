import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    Funziona su batch di embedding e coppie positive.
    """
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        """
        Crea una matrice [2*B, 2*B] che maschera le coppie positive.
        """
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # esclude se stessa
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = 0
            mask[i + self.batch_size, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: embedding di shape [B, D] (dopo projection head)
        Ritorna la loss scalare NT-Xent
        """
        # 1. Normalizza embedding
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2. Concatenazione (2B, D)
        representations = torch.cat([z_i, z_j], dim=0)

        # 3. Matrice di similarità (cosine similarity)
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        # 4. Exclude positive pairs from denominator using mask
        sim_matrix_exp = torch.exp(sim_matrix) * self.mask

        # 5. Positive similarity (diagonal opposta)
        positives = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        positives = torch.cat([positives, positives], dim=0)

        # 6. Loss
        loss = -torch.log(positives / sim_matrix_exp.sum(dim=1))
        loss = loss.mean()
        return loss
