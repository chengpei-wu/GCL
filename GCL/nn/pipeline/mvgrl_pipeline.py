import dgl
import torch
import torch.nn as nn

from GCL.augmentation.personalized_page_rank import PersonalizedPageRankAug
from GCL.contrastive_loss.jsd import JSD
from GCL.contrastive_manager.mvgrl_contrast import MVGRLContrast
from GCL.nn.encoder.gcn_encoder import GCNEncoder


class MVGRLPipeline(nn.Module):
    def __init__(self, feat_size: int, embed_dim: int, task_level='graph'):
        super(MVGRLPipeline, self).__init__()
        self.task_level = task_level
        self.augmentor = PersonalizedPageRankAug()
        self.encoder = GCNEncoder(in_size=feat_size, embed_size=embed_dim, num_layers=4, level='both', pool='sum')
        self.contrast = MVGRLContrast(embed_dim=embed_dim, loss=JSD(), task_level='graph')

    def forward_graph(self, batch_graph: dgl.DGLGraph, feat='feat'):
        diff_batch_graph = self.augmentor(batch_graph)
        local_h1, global_h1 = self.encoder(batch_graph, batch_graph.ndata[feat])
        local_h2, global_h2 = self.encoder(diff_batch_graph, batch_graph.ndata[feat])

        batch_graph_id = dgl.broadcast_nodes(batch_graph, torch.arange(batch_graph.batch_size).to(local_h1.device))

        loss = self.contrast(local_h1, local_h2, global_h1, global_h2, batch_graph_id)
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
    ...
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # load dataset
    # dataset = GINDataset('MUTAG', self_loop=True)
    # dataloader = GraphDataLoader(dataset, batch_size=64)
    #
    # # MVGRL pipeline
    # mvgrl = MVGRLPipeline(1, 32, task_level='graph').to(device)
    #
    # # MVGRL training
    # optimizer = torch.optim.Adam(mvgrl.parameters(), lr=1e-2)
    # for epoch in range(100):
    #     epoch_losses = []
    #     for batched_graph, _ in dataloader:
    #         optimizer.zero_grad()
    #
    #         batched_graph = batched_graph.to(device)
    #         batched_graph.ndata['feat'] = batched_graph.in_degrees().float().view(-1, 1).to(device)
    #
    #         loss = mvgrl(batched_graph, 'feat')
    #
    #         loss.backward()
    #         optimizer.step()
    #         epoch_losses.append(loss.item())
    #
    #     print(f'epoch: {epoch}, training loss: {np.mean(epoch_losses)}')
    #
    # # test MVGRL-learned graph-level embeddings using SVM classifier
    # evaluator = SVCEvaluator(measures=['accuracy'])
    #
    # graphs, labels = map(list, zip(*dataset))
    # whole_batch_graph = dgl.batch(graphs).to(device)
    # whole_batch_graph.ndata['feat'] = whole_batch_graph.in_degrees().float().view(-1, 1).to(device)
    #
    # embeddings = mvgrl.get_embedding(whole_batch_graph, 'feat').detach().cpu()
    # labels = torch.tensor(labels).squeeze()
    #
    # print(labels.shape, embeddings.shape)
    #
    # # here, to obtain fair results, better using 10-fold cv, and report the average acc.
    # masks = train_test_split(embeddings.size(0), valid_size=0.1, test_size=0.1)
    # best_performance = evaluator(embeddings, labels, masks=masks)
    # print(best_performance)
