import dgl
from dgl import DropNode

from GCL.augmentation.augmentor import Augmentor


class DropNodeAug(Augmentor):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.drop = DropNode(prob)

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        return self.drop(graph)


if __name__ == '__main__':
    g = dgl.rand_graph(20, 40)
    aug = DropNodeAug(0.5)
    new_g = aug.augment(g)
    print(new_g.nodes(), new_g.num_nodes())
    print(g.nodes(), g.num_nodes())
