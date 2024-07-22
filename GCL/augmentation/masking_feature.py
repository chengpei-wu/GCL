import dgl
import torch
from dgl import FeatMask

from GCL.augmentation.augmentor import Augmentor


class MaskingFeatureAug(Augmentor):
    def __init__(self, feature_field, prob=0.5):
        super().__init__()
        self.p = prob
        self.feature_field = feature_field
        self.transform = FeatMask(p=prob, node_feat_names=[feature_field])

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return self.transform(graph)


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    g.ndata['feat'] = torch.randn(20, 40)
    aug = MaskingFeatureAug('feat', 0.5)
    new_g = aug(g)
    print(g.ndata['feat'])
    print(new_g.ndata['feat'])
