import dgl
import torch.nn as nn

from GCL.augmentation.shuffle_node import ShuffleNodeAug
from GCL.contrastive_manager.dgi_contrast import DGIContrast
from GCL.nn.encoder.gcn_encoder import GCNEncoder


class DGIPipeline(nn.Module):
    def __init__(self, feat_size: int, embed_dim: int):
        super(DGIPipeline, self).__init__()
        self.augmentor = ShuffleNodeAug()
        self.encoder = GCNEncoder(in_size=feat_size, embed_size=embed_dim, num_layers=1)
        self.contrast = DGIContrast(embed_dim=embed_dim, loss=nn.BCEWithLogitsLoss())

    def forward(self, graph: dgl.DGLGraph, feat='feat'):
        g1 = graph.clone()
        g2 = self.augmentor(graph)

        h1 = self.encoder(g1, g1.ndata[feat])

        h2 = self.encoder(g2, g2.ndata[feat])

        loss = self.contrast(h1, h2)

        return loss

    def get_embedding(self, graph: dgl.DGLGraph, feat='feat'):
        return self.encoder(graph, graph.ndata[feat])


if __name__ == '__main__':
    ...
