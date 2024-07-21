import dgl
import numpy as np
import torch
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader

from GCL.embedding_evaluator.evaluator import SVCEvaluator
from GCL.embedding_evaluator.utils import train_test_split
from GCL.nn.infograph_pipeline import InfoGraphPipeline


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = GINDataset('IMDBBINARY', self_loop=True)
    dataloader = GraphDataLoader(dataset, batch_size=128)

    # InfoGraph pipeline
    InfoGraph = InfoGraphPipeline(1, 32).to(device)

    # InfoGraph training
    optimizer = torch.optim.Adam(InfoGraph.parameters(), lr=1e-2)
    for epoch in range(100):
        epoch_losses = []
        for batched_graph, _ in dataloader:
            optimizer.zero_grad()

            batched_graph = batched_graph.to(device)
            batched_graph.ndata['feat'] = batched_graph.in_degrees().float().view(-1, 1).to(device)

            loss = InfoGraph(batched_graph, 'feat')

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'epoch: {epoch}, training loss: {np.mean(epoch_losses)}')

    # test InfoGraph-learned graph-level embeddings using SVM classifier
    evaluator = SVCEvaluator(measures=['accuracy'])

    graphs, labels = map(list, zip(*dataset))
    whole_batch_graph = dgl.batch(graphs).to(device)
    whole_batch_graph.ndata['feat'] = whole_batch_graph.in_degrees().float().view(-1, 1).to(device)

    embeddings = InfoGraph.get_embedding(whole_batch_graph, 'feat').detach().cpu()
    labels = torch.tensor(labels).squeeze()

    print(labels.shape, embeddings.shape)

    # here, to obtain fair results, better using 10-fold cv, and report the average acc.
    masks = train_test_split(embeddings.size(0), valid_size=0.1, test_size=0.1)
    best_performance = evaluator(embeddings, labels, masks=masks)
    print(best_performance)