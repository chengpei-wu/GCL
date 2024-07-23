from copy import deepcopy

import dgl
import torch
import torch.nn.functional as F

from GCL.augmentation.augmentor import Augmentor


class DropoutFeatureAug(Augmentor):
    def __init__(self, feature_field, p=0.5):
        super().__init__()
        self.p = p
        self.feature_field = feature_field

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        g = deepcopy(graph)

        features = g.ndata[self.feature_field]

        x = F.dropout(features, self.p)
        g.ndata[self.feature_field] = x
        return g


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    g.ndata['feat'] = torch.randn(20, 40)
    aug = DropoutFeatureAug('feat', 0.5)
    new_g = aug(g)
    print(g.ndata['feat'])
    print(new_g.ndata['feat'])
