import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import SumPooling

from GCL.contrastive_loss.jsd import JSD


class MVGRLDiscriminator(nn.Module):
    def __init__(self, h_dim):
        super(MVGRLDiscriminator, self).__init__()
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


class MVGRLContrast(nn.Module):
    def __init__(self, embed_dim: int, loss: nn.Module = JSD(), task_level='graph'):
        super(MVGRLContrast, self).__init__()
        self.loss_func = loss
        self.local_discriminator = MVGRLDiscriminator(embed_dim)
        self.global_discriminator = MVGRLDiscriminator(embed_dim)
        self.task_level = task_level

    def forward(self, *args) -> torch.Tensor:
        if self.task_level == 'graph':
            return self.forward_graph(*args)
        elif self.task_level == 'node':
            return self.forward_node()
        else:
            raise NotImplementedError

    def forward_node(self) -> torch.Tensor:
        ...

    def forward_graph(self, h1: torch.Tensor, h2: torch.Tensor, g_h1: torch.Tensor, g_h2: torch.Tensor,
            batch_graph_id: torch.Tensor) -> torch.Tensor:
        device = h1.device

        lh1 = self.local_discriminator(h1)
        lh2 = self.local_discriminator(h2)

        gh1 = self.global_discriminator(g_h1)
        gh2 = self.global_discriminator(g_h2)

        pos_mask = torch.zeros((h1.size(0), g_h1.size(0))).to(device)
        neg_mask = torch.ones((h1.size(0), g_h1.size(0))).to(device)

        for i in range(g_h1.size(0)):
            pos_mask[:, i][batch_graph_id == i] = 1.0
            neg_mask[:, i][batch_graph_id == i] = 0.0

        similarity_score1 = torch.mm(lh1, gh2.t())
        similarity_score2 = torch.mm(lh2, gh1.t())

        pos_score1 = (similarity_score1 * pos_mask).view(-1, 1)
        neg_score1 = (similarity_score1 * neg_mask).view(-1, 1)

        pos_score2 = (similarity_score2 * pos_mask).view(-1, 1)
        neg_score2 = (similarity_score2 * neg_mask).view(-1, 1)

        return self.loss_func(pos_score1, neg_score1) + self.loss_func(pos_score2, neg_score2)


if __name__ == '__main__':
    from GCL.augmentation.personalized_page_rank import PersonalizedPageRankAug

    g1 = dgl.rand_graph(10, 20)
    g2 = dgl.rand_graph(10, 20)
    batch_g = dgl.batch([g1, g2])
    feats = torch.randn((20, 16))

    aug = PersonalizedPageRankAug()
    diff_batch_g = aug(batch_g)

    batch_graph_id = dgl.broadcast_nodes(batch_g, torch.arange(2))

    print(batch_graph_id)
    c = MVGRLContrast(16)

    readout = SumPooling()
    loss = c(feats, feats, readout(batch_g, feats), readout(diff_batch_g, feats), batch_graph_id)
    print(loss)
