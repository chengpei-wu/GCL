import dgl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from GCL.augmentation.augmentor import Compose
from GCL.augmentation.drop_edge import DropEdgeAug
from GCL.augmentation.masking_feature import MaskingFeatureAug
from GCL.contrastive_loss.info_nce import InfoNCE
from GCL.contrastive_manager.grace_contrast import GraceContrast
from GCL.embedding_evaluator.evaluator import LREvaluator
from GCL.nn.gcn_encoder import GCNEncoder


class GracePipeline(nn.Module):
    def __init__(
            self, feat_size: int, embed_dim: int, proj_dim: int,
            p_d1: float, p_m1: float, p_d2: float, p_m2: float, tau: float
    ):
        super(GracePipeline, self).__init__()
        self.augmentor1 = Compose([DropEdgeAug(prob=p_d1), MaskingFeatureAug('feat', prob=p_m1)])
        self.augmentor2 = Compose([DropEdgeAug(prob=p_d2), MaskingFeatureAug('feat', prob=p_m2)])
        self.encoder = GCNEncoder(in_size=feat_size, embed_size=embed_dim, num_layers=2)
        self.contrast = GraceContrast(embed_dim=embed_dim, proj_dim=proj_dim, loss=InfoNCE(tau=tau))

    def forward(self, graph: dgl.DGLGraph, feat='feat'):
        g1 = self.augmentor1(graph)
        g2 = self.augmentor2(graph)

        h1 = self.encoder(g1, g1.ndata[feat])
        h2 = self.encoder(g2, g2.ndata[feat])

        loss = self.contrast(h1, h2)

        return loss

    def get_embedding(self, graph: dgl.DGLGraph, feat='feat'):
        return self.encoder(graph, graph.ndata[feat]).detach()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = dgl.data.CoraGraphDataset()
    num_classes = dataset.num_classes
    g = dataset[0].to(device)
    g = dgl.add_self_loop(g)
    Grace = GracePipeline(
        feat_size=g.ndata["feat"].shape[1], embed_dim=256, proj_dim=1024,
        p_d1=0.2, p_m1=0.2, p_d2=0.2, p_m2=0.2, tau=0.5
    ).to(device)

    # train GRACE
    optimizer = torch.optim.Adam(Grace.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(10000):
        optimizer.zero_grad()

        loss = Grace(g)
        loss.backward()

        optimizer.step()
        print(f'epoch: {epoch}, training loss: {loss.item()}')

    # test GRACE embedding
    embedding = Grace.get_embedding(g)
    print(embedding)
    evaluator = LREvaluator(measures=['macro_f1'])
    masks = {
        'train': g.ndata["train_mask"],
        'valid': g.ndata["val_mask"],
        'test':  g.ndata["test_mask"],
    }
    # masks = train_test_split(embedding.size(0), test_size=0.8)
    _, best_performance = evaluator(embedding, g.ndata["label"], masks=masks)
    print(best_performance)
