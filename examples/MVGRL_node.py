import dgl
import torch

from GCL.embedding_evaluator.evaluator import LREvaluator
from GCL.nn.pipeline.mvgrl_pipeline import MVGRLPipeline


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
