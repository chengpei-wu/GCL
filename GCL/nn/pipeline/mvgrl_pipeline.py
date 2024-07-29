import dgl
import torch
import torch.nn as nn

from GCL.augmentation.personalized_page_rank import PersonalizedPageRankAug
from GCL.augmentation.shuffle_node import ShuffleNodeAug
from GCL.contrastive_loss.jsd import JSD
from GCL.contrastive_manager.mvgrl_contrast import MVGRLContrast
from GCL.embedding_evaluator.evaluator import LREvaluator
from GCL.nn.encoder.gcn_encoder import GCNEncoder


class MVGRLPipeline(nn.Module):
    def __init__(self, feat_size: int, embed_dim: int, task_level='graph'):
        super(MVGRLPipeline, self).__init__()
        self.task_level = task_level
        self.augmentor = PersonalizedPageRankAug()
        self.encoder = GCNEncoder(in_size=feat_size, embed_size=embed_dim, num_layers=4, level='both', pool='sum')
        if task_level == 'graph':
            self.contrast = MVGRLContrast(embed_dim=embed_dim, loss=JSD(), task_level='graph')
        else:
            self.contrast = MVGRLContrast(embed_dim=embed_dim, loss=nn.BCEWithLogitsLoss(), task_level='node')

    def forward_graph(self, batch_graph: dgl.DGLGraph, feat='feat'):
        diff_batch_graph = self.augmentor(batch_graph)
        local_h1, global_h1 = self.encoder(batch_graph, batch_graph.ndata[feat])
        local_h2, global_h2 = self.encoder(diff_batch_graph, batch_graph.ndata[feat])

        batch_graph_id = dgl.broadcast_nodes(batch_graph, torch.arange(batch_graph.batch_size).to(local_h1.device))

        loss = self.contrast(local_h1, local_h2, global_h1, global_h2, batch_graph_id)
        return loss

    def forward_node(self, graph: dgl.DGLGraph, feat='feat'):
        s = ShuffleNodeAug()
        diff_graph = self.augmentor(graph)

        h1, c1 = self.encoder(graph, graph.ndata[feat])
        h2, c2 = self.encoder(graph, graph.ndata[feat])

        node_shuffle_graph1 = s(graph)
        node_shuffle_graph2 = s(diff_graph)

        h3, _ = self.encoder(node_shuffle_graph1, node_shuffle_graph1.ndata[feat])
        h4, _ = self.encoder(node_shuffle_graph2, node_shuffle_graph2.ndata[feat])

        loss = self.contrast(h1, h2, h3, h4, c1, c2)
        return loss


    def forward(self, *args):
        if self.task_level == 'graph':
            return self.forward_graph(*args)
        elif self.task_level == 'node':
            return self.forward_node(*args)

    def get_embedding(self, graph: dgl.DGLGraph, feat='feat', level='graph'):
        node_h, graph_h = self.encoder(graph, graph.ndata[feat])
        if level == 'node':
            return node_h
        elif level == 'graph':
            return graph_h
        else:
            raise NotImplementedError


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = dgl.data.CoraGraphDataset()
    num_classes = dataset.num_classes
    g = dataset[0].to(device)
    mvgrl = MVGRLPipeline(
        feat_size=g.ndata["feat"].shape[1], embed_dim=128, task_level='node'
    ).to(device)

    # train MVGRL
    optimizer = torch.optim.Adam(mvgrl.parameters(), lr=0.0005, weight_decay=1e-5)

    for epoch in range(80):
        optimizer.zero_grad()

        loss = mvgrl(g, 'feat')
        loss.backward()

        optimizer.step()
        print(f'epoch: {epoch}, training loss: {loss.item()}')

    # test MVGRL embedding
    g = dgl.add_self_loop(g)
    embedding = mvgrl.get_embedding(g, 'feat', 'node')
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
