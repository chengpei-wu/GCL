import torch
from torch import nn

from GCL.contrastive_loss.info_nce import InfoNCE


class GraceProjector(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super(GraceProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ELU(),
            nn.Linear(proj_dim, in_dim),
        )

    def forward(self, h):
        z = self.mlp(h)
        return z


class GraceContrast(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int, loss: nn.Module):
        super(GraceContrast, self).__init__()
        self.loss_func = loss
        self.projector = GraceProjector(embed_dim, proj_dim)

    def forward(self, h1, h2):
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss1 = self.loss_func(z1, z2)
        loss2 = self.loss_func(z2, z1)

        return (loss1 + loss2) * 0.5


if __name__ == '__main__':
    h1 = torch.randn((4, 16))
    h2 = torch.randn((4, 16))

    c = GraceContrast(16, 16, InfoNCE(tau=0.5))
    loss = c(h1, h2)
    # print(loss)
