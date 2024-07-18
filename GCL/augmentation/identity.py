import dgl
import torch

from GCL.augmentation.augmentor import Augmentor


class IdentityAug(Augmentor):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return graph.clone()


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    g.ndata['feat'] = torch.randn(20, 40)
    aug = IdentityAug()
    new_g = aug(g)
    print(g.ndata['feat'])
    print(new_g.ndata['feat'])
