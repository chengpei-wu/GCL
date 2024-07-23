from copy import deepcopy

import dgl
import dgl.sparse as dglsp
import torch

from GCL.augmentation.augmentor import Augmentor


class PersonalizedPageRank(Augmentor):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        device = graph.device
        res_g = deepcopy(graph)

        adj = graph.adj()  # a dgl sparse matrix
        d = dglsp.diag(adj.sum(dim=1))  # a dgl sparse matrix

        d_inv = d ** -0.5
        norm_adj = d_inv @ adj @ d_inv
        identity = dglsp.identity(adj.shape)
        ppr_matrix = self.alpha * (identity - (1 - self.alpha) * norm_adj) ** -1

        new_edges = ppr_matrix.indices()
        edge_ids = res_g.edge_ids(*res_g.edges())
        res_g.remove_edges(edge_ids)
        res_g.add_edges(new_edges[0], new_edges[1])
        res_g.edata['edge_weight'] = ppr_matrix.val

        res_g.to(device)
        return res_g


if __name__ == '__main__':
    g = dgl.rand_graph(10, 20)
    g.ndata['feat'] = torch.randn((10, 20))
    ppr = PersonalizedPageRank()
    new_g = ppr(g)

    print(g)
    print(new_g)
