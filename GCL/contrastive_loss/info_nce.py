import torch
import torch.nn as nn

import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1, z2 = F.normalize(z1), F.normalize(z2)

    intra_sim = torch.exp(torch.mm(z1, z1.t()) / tau)
    inter_sim = torch.exp(torch.mm(z1, z2.t()) / tau)

    denominator = intra_sim.sum(dim=1) + inter_sim.sum(dim=1) - intra_sim.diag()
    numerator = inter_sim.diag()

    loss = -torch.log(numerator / denominator).mean()
    return loss


class InfoNCE(nn.Module):
    def __init__(self, tau: float):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def __call__(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return info_nce_loss(z1, z2, self.tau)
