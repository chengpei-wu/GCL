from abc import ABC, abstractmethod
from typing import List

import dgl


class Augmentor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        raise NotImplementedError('')

    def __call__(self, graph):
        return self.augment(graph)


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        for aug in self.augmentors:
            graph = aug.augment(graph)
        # graph = dgl.add_self_loop(graph)
        return graph
