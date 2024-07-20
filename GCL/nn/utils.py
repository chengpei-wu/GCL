import dgl.nn.pytorch
from torch import nn


def readout_map(pool: str) -> nn.Module:
    if pool == 'mean':
        return dgl.nn.pytorch.AvgPooling()
    elif pool == 'max':
        return dgl.nn.pytorch.MaxPooling()
    elif pool == 'sum':
        return dgl.nn.pytorch.SumPooling()
    else:
        raise NotImplementedError(pool)
