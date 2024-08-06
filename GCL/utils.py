import dgl
import numpy as np
import pymetis as metis
import torch


def metis_partition(g: dgl.DGLGraph, n_cluster: int) -> tuple:
    adj_list = [[] for _ in range(g.number_of_nodes())]
    edges = g.edges()
    for u, v in zip(edges[0].tolist(), edges[1].tolist()):
        adj_list[u].append(v)
        adj_list[v].append(u)

    cuts, membership = metis.part_graph(n_cluster, adj_list, recursive=True)
    membership = np.array(membership)

    cluster_graphs = [dgl.node_subgraph(graph=g, nodes=np.where(membership == c)[0]) for c in range(n_cluster)]

    return cuts, dgl.batch(cluster_graphs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = dgl.data.CoraGraphDataset()
    num_classes = dataset.num_classes
    g = dataset[0].to(device)
    g = dgl.add_self_loop(g)

    cuts, cluster_batch_graphs = metis_partition(g, n_cluster=2)

    print(cuts, cluster_batch_graphs)
