from copy import deepcopy

import dgl
import dgl.sparse as dglsp

from GCL.augmentation.augmentor import Augmentor


class PersonalizedPageRankAug(Augmentor):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        device = graph.device
        res_g = deepcopy(graph).to(device)

        adj = graph.adj()  # a dgl sparse matrix
        d = dglsp.diag(adj.sum(dim=1)).to(device)  # a dgl sparse matrix

        d_inv = d ** -0.5
        norm_adj = d_inv @ adj @ d_inv
        identity = dglsp.identity(adj.shape).to(device)
        ppr_matrix = self.alpha * (identity - (1 - self.alpha) * norm_adj) ** -1

        new_edges = ppr_matrix.indices()
        edge_ids = res_g.edge_ids(*res_g.edges())
        res_g.remove_edges(edge_ids)
        res_g.add_edges(new_edges[0], new_edges[1])
        res_g.edata['edge_weight'] = ppr_matrix.val

        res_g.to(device)
        return res_g


if __name__ == '__main__':
    ...
    # def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    #     a = nx.convert_matrix.to_numpy_array(graph)
    #     if self_loop:
    #         a = a + np.eye(a.shape[0])  # A^ = A + I_n
    #     d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    #     dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    #     at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    #     return alpha * inv(
    #         (np.eye(a.shape[0]) - (1 - alpha) * at)
    #     )
    #
    #
    # dataset = dgl.data.CoraGraphDataset()
    # num_classes = dataset.num_classes
    # g = dataset[0]
    #
    # g = dgl.to_networkx(g)
    # lap = nx.normalized_laplacian_matrix(g)
    # nx.normalized_laplacian_spectrum
    # ppr = compute_ppr(g)
    # print(ppr)
