import dgl
import torch
from dgl import NodeShuffle

from GCL.augmentation.augmentor import Augmentor


class ShuffleNodeAug(Augmentor):
    def __init__(self):
        super().__init__()
        self.transform = NodeShuffle()

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return self.transform(graph)


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    g.ndata['feat'] = torch.arange(100).view(20, -1)
    aug = ShuffleNodeAug()
    new_g = aug(g)
    print(g.ndata['feat'])
    print(new_g.ndata['feat'])
