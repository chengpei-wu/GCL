import dgl
import torch
import torch.nn as nn

from GCL.augmentation.shuffle_node import ShuffleNodeAug
from GCL.contrastive_manager.dgi_contrast import DGIContrast
from GCL.nn.gcn_encoder import GCNEncoder


class DGIPipeline(nn.Module):
    def __init__(self, feat_size: int, embed_dim: int):
        super(DGIPipeline, self).__init__()
        self.augmentor = ShuffleNodeAug()
        self.encoder = GCNEncoder(in_size=feat_size, hid_size=embed_dim, num_layers=1)
        self.contrast = DGIContrast(h_dim=embed_dim, loss=nn.BCEWithLogitsLoss())

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = dgl.data.CoraGraphDataset()
    num_classes = dataset.num_classes
    g = dataset[0].to(device)
    g = dgl.add_self_loop(g)
    DGI = DGIPipeline(feat_size=g.ndata["feat"].shape[1], embed_dim=512).to(device)

    # train DGI
    optimizer = torch.optim.Adam(DGI.parameters(), lr=1e-2)
    # for epoch in range(500):
    #     optimizer.zero_grad()
    #     loss = DGI(g)
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())
    #
    # # test DGI embedding
    # embedding = DGI.get_embedding(g)
    # evaluator = LREvaluator(measures=['accuracy'])
    # masks = {
    #     'train': g.ndata["train_mask"],
    #     'valid': g.ndata["val_mask"],
    #     'test':  g.ndata["test_mask"],
    # }
    # # masks = train_test_split(embedding.size(0), test_size=0.8)
    # _, best_performance = evaluator(embedding, g.ndata["label"], masks=masks)
    # print(best_performance)
