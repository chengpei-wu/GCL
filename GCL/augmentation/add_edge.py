import dgl
import torch
from dgl import AddEdge

from GCL.augmentation.augmentor import Augmentor


class AddEdgeAug(Augmentor):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.transform = AddEdge(ratio)

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return self.transform(graph)


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    aug = AddEdgeAug(0.1)
    new_g = aug(g)
    print(torch.vstack([*new_g.edges()]), new_g.num_edges())
    print(torch.vstack([*g.edges()]), g.num_edges())
