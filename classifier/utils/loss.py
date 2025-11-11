import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, tau_plus=0.1, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.eps = eps

    def forward(self, z_i, z_j):
        B = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        sim = torch.matmul(representations, representations.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        mask = torch.eye(2*B, dtype=torch.bool, device=z_i.device)
        sim.masked_fill_(mask, -9e15)

        positives = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        positives = torch.cat([positives, positives], dim=0)

        neg_sum = torch.exp(sim).sum(dim=1)
        Ng = (-self.tau_plus * B * positives + neg_sum) / (1 - self.tau_plus)
        Ng = torch.clamp(Ng, min=B * torch.exp(torch.tensor(-1/self.temperature)))

        loss = -torch.log(positives / (positives + Ng + self.eps))
        return loss.mean()
