import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn

from GCL.nn.utils import readout_map


class GCNEncoder(nn.Module):
    def __init__(self, in_size, embed_size=16, num_layers=1, activation=F.relu, level='node',
            pool: str = 'mean'):
        super(GCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.rep_level = level
        if self.rep_level in ['graph', 'both']:
            self.readout = readout_map(pool)

        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GraphConv(
                        in_size, embed_size, allow_zero_in_degree=True, activation=activation
                    )
                )
            else:
                self.layers.append(
                    GraphConv(
                        embed_size, embed_size, allow_zero_in_degree=True, activation=activation
                    )
                )

    def forward(self, graph: dgl.DGLGraph, feats: torch.Tensor, edge_weight=None):
        h = feats
        for layer in self.layers:
            h = layer(graph, h, edge_weight=edge_weight)
        if self.rep_level == 'node':
            return h
        elif self.rep_level == 'graph':
            return self.readout(graph, h)
        elif self.rep_level == 'both':
            return h, self.readout(graph, h)
        else:
            raise NotImplementedError(self.rep_level)


if __name__ == "__main__":
    g = dgl.rand_graph(20, 40)
    g.ndata['feats'] = torch.randn((20, 40))

    encoder = GCNEncoder(40, 512, 1, level='graph', pool='mean')

    h = encoder(g, g.ndata['feats'])
    print(h.shape)
