import dgl
import torch
from dgl import DropEdge

from GCL.augmentation.augmentor import Augmentor


class DropEdgeAug(Augmentor):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.drop = DropEdge(prob)

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return self.drop(graph)


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    aug = DropEdgeAug(0.1)
    new_g = aug.augment(g)
    print(torch.vstack([*new_g.edges()]), new_g.num_edges())
    print(torch.vstack([*g.edges()]), g.num_edges())
