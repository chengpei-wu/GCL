import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, h_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Bilinear(h_dim, h_dim, 1)

    def forward(self, s, h):
        s = s.expand_as(h)
        similarity = self.weight(h, s)
        return similarity


class DGIContrast(nn.Module):
    def __init__(self, h_dim: int, loss: nn.Module):
        super(DGIContrast, self).__init__()
        self.loss_func = loss
        self.discriminator = Discriminator(h_dim)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        device = h1.device

        s = torch.mean(h1, dim=0)

        pos_label = torch.ones(h1.size(0), 1).to(device)
        neg_label = torch.zeros(h2.size(0), 1).to(device)
        label = torch.cat((pos_label, neg_label), 0)

        pos_logits = self.discriminator(s, h1)
        neg_logits = self.discriminator(s, h2)

        logits = torch.cat((pos_logits, neg_logits), 0)

        loss = self.loss_func(logits, label)

        return loss


if __name__ == '__main__':
    h1 = torch.randn((4, 16))
    h2 = torch.randn((4, 16))

    c = DGIContrast(16, nn.BCEWithLogitsLoss())
    loss = c(h1, h2)
    print(loss)
