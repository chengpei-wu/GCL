import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import SumPooling

from GCL.contrastive_loss.jsd import JSD


class Discriminator(nn.Module):
    def __init__(self, h_dim):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.jump_con = nn.Linear(h_dim, h_dim)

    def forward(self, x):
        block_out = self.block(x)
        jump_out = self.jump_con(x)
        out = block_out + jump_out
        return out


class InfoGraphContrast(nn.Module):
    def __init__(self, h_dim: int, loss: nn.Module = JSD()):
        super(InfoGraphContrast, self).__init__()
        self.loss_func = loss
        self.local_discriminator = Discriminator(h_dim)
        self.global_discriminator = Discriminator(h_dim)

    def forward(self, h: torch.Tensor, g_h: torch.Tensor, batch_graph_id: torch.Tensor) -> torch.Tensor:
        device = h.device

        lh = self.local_discriminator(h)
        gh = self.global_discriminator(g_h)

        pos_mask = torch.zeros((h.size(0), g_h.size(0))).to(device)
        neg_mask = torch.ones((h.size(0), g_h.size(0))).to(device)

        for i in range(g_h.size(0)):
            pos_mask[:, i][batch_graph_id == i] = 1.0
            neg_mask[:, i][batch_graph_id == i] = 0.0

        # for nodeidx, graphidx in enumerate(batch_graph_id):
        #     pos_mask[nodeidx][graphidx] = 1.0
        #     neg_mask[nodeidx][graphidx] = 0.0

        similarity_score = torch.mm(lh, gh.t())

        pos_score = (similarity_score * pos_mask).view(-1, 1)
        neg_score = (similarity_score * neg_mask).view(-1, 1)

        return self.loss_func(pos_score, neg_score)


if __name__ == '__main__':
    g1 = dgl.rand_graph(10, 20)
    g2 = dgl.rand_graph(10, 20)
    batch_g = dgl.batch([g1, g2])
    feats = torch.randn((20, 16))
    batch_graph_id = dgl.broadcast_nodes(batch_g, torch.arange(2))

    print(batch_graph_id)
    c = InfoGraphContrast(16)

    readout = SumPooling()
    loss = c(feats, readout(batch_g, feats), batch_graph_id)
    print(loss)
