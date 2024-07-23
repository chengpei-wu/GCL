from copy import deepcopy

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
        return self.transform(deepcopy(graph))


if __name__ == '__main__':
    feat_mask = FeatMask(0.5, ['feat'])
    g = dgl.rand_graph(10, 20)
    g.ndata['feat'] = torch.randn(10, 5)

    for i in range(10):
        new_g = feat_mask(deepcopy(g))

        print(new_g.ndata['feat'])
        print(g.ndata['feat'])
