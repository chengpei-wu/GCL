from abc import ABC, abstractmethod

import dgl
import torch


class Loss(ABC):
    @abstractmethod
    def compute(self, graph: dgl.DGLGraph) -> torch.FloatTensor:
        pass

    def __call__(self, graph: dgl.DGLGraph) -> torch.FloatTensor:
        loss = self.compute(graph)
        return loss
