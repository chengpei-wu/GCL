import dgl
import torch
import torch.nn as nn

from GCL.contrastive_loss.jsd import JSD
from GCL.contrastive_manager.info_graph_contrast import InfoGraphContrast
from GCL.nn.gin_encoder import GINEncoder


class InfoGraphPipeline(nn.Module):
    def __init__(self, feat_size: int, embed_dim: int):
        super(InfoGraphPipeline, self).__init__()
        self.encoder = GINEncoder(in_size=feat_size, hid_size=embed_dim, num_layers=1)
        self.contrast = InfoGraphContrast(embed_dim=embed_dim, loss=JSD())

    def forward(self, batch_graph: dgl.DGLGraph, feat='feat'):
        global_h, local_h = self.encoder(batch_graph, batch_graph.ndata[feat])
        batch_graph_id = dgl.broadcast_nodes(batch_graph, torch.arange(batch_graph.batch_size).to(global_h.device))

        loss = self.contrast(local_h, global_h, batch_graph_id)
        return loss

    def get_embedding(self, graph: dgl.DGLGraph, feat='feat'):
        graph_h, _ = self.encoder(graph, graph.ndata[feat])
        return graph_h
