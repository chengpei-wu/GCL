import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv
from torch import nn

from GCL.nn.utils import readout_map


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GINEncoder(nn.Module):
    def __init__(self, in_size, hid_size=32, num_layers=3, activation=F.relu, level='graph',
            pool: str = 'sum'):
        super(GINEncoder, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.rep_level = level
        if self.rep_level == 'graph':
            self.readout = readout_map(pool)
        self.num_layers = num_layers
        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(in_size, hid_size, hid_size)
            else:
                mlp = MLP(hid_size, hid_size, hid_size)

            self.gin_layers.append(GINConv(mlp, aggregator_type='sum', activation=activation))
            self.batch_norms.append(nn.BatchNorm1d(hid_size))

    def forward(self, graph: dgl.DGLGraph, feats: torch.Tensor) -> tuple:
        h = feats
        hs = []
        for i in range(self.num_layers):
            h = self.gin_layers[i](graph, h)
            h = self.batch_norms[i](h)
            hs.append(h)

        node_h = torch.cat(hs, 1)
        graph_h = self.readout(graph, node_h)

        return graph_h, node_h


if __name__ == "__main__":
    g = dgl.rand_graph(20, 40)
    g.ndata['feats'] = torch.randn((20, 40))

    encoder = GINEncoder(40, 32, 3, level='graph', pool='sum')

    g_h, h = encoder(g, g.ndata['feats'])
    print(g_h.shape, h.shape)
