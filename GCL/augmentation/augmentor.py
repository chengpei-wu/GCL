from abc import ABC, abstractmethod

import dgl


class Augmentor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        raise NotImplementedError('')

    def __call__(self, graph):
        return self.augment(graph)
