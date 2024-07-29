import dgl
import torch

from GCL.embedding_evaluator.evaluator import LREvaluator
from GCL.nn.pipeline.grace_pipeline import GracePipeline


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = dgl.data.CoraGraphDataset()
    num_classes = dataset.num_classes
    g = dataset[0].to(device)
    Grace = GracePipeline(
        feat_size=g.ndata["feat"].shape[1], embed_dim=256, proj_dim=256,
        p_d1=0.2, p_m1=0.2, p_d2=0.2, p_m2=0.2, tau=1
    ).to(device)

    # train GRACE
    optimizer = torch.optim.Adam(Grace.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(1000):
        optimizer.zero_grad()

        loss = Grace(g)
        loss.backward()

        optimizer.step()
        print(f'epoch: {epoch}, training loss: {loss.item()}')

    # test GRACE embedding
    g = dgl.add_self_loop(g)
    embedding = Grace.get_embedding(g)
    print(embedding)
    evaluator = LREvaluator(measures=['accuracy'])

    masks = {
        'train': g.ndata["train_mask"],
        'valid': g.ndata["val_mask"],
        'test':  g.ndata["test_mask"],
    }
    print(masks['train'].sum(), masks['valid'].sum(), masks['test'].sum())
    _, best_performance = evaluator(embedding, g.ndata["label"], masks=masks)
    print(best_performance)
