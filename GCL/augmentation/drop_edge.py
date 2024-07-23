import dgl
import torch
from dgl import DropEdge

from GCL.augmentation.augmentor import Augmentor


class DropEdgeAug(Augmentor):
    def __init__(self, prob=0.5):
        super().__init__()
        self.transform = DropEdge(prob)

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return self.transform(graph)


if __name__ == '__main__':
    graph = dgl.rand_graph(20, 40)
    aug = DropEdgeAug(1)
    for i in range(10):
        new_g = aug(graph)
        print(new_g.num_edges())
        print(graph.num_edges())
