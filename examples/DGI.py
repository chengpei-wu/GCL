import dgl
import torch

from GCL.embedding_evaluator.evaluator import LREvaluator
from GCL.nn.dgi_pipeline import DGIPipeline


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = dgl.data.CoraGraphDataset()
    num_classes = dataset.num_classes
    g = dataset[0].to(device)
    g = dgl.add_self_loop(g)

    # DGI pipeline
    DGI = DGIPipeline(feat_size=g.ndata["feat"].shape[1], embed_dim=512).to(device)

    # DGI training
    optimizer = torch.optim.Adam(DGI.parameters(), lr=1e-2)
    for epoch in range(500):
        optimizer.zero_grad()
        loss = DGI(g)
        loss.backward()
        optimizer.step()
        print(f'epoch: {epoch}, training loss: {loss.item()}')

    # test DGI-learned node embedding on Logistic Regression model using transductive setting
    embedding = DGI.get_embedding(g)
    evaluator = LREvaluator(measures=['accuracy'])
    masks = {
        'train': g.ndata["train_mask"],
        'valid': g.ndata["val_mask"],
        'test':  g.ndata["test_mask"],
    }
    # masks = train_test_split(embedding.size(0), test_size=0.8)
    _, best_performance = evaluator(embedding, g.ndata["label"], masks=masks)
    print(best_performance)
