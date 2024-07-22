import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def jsd_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    pos_loss = (np.log(2) - F.softplus(-positive_scores)).mean()
    neg_loss = (F.softplus(-negative_scores) + negative_scores - np.log(2)).mean()
    return neg_loss - pos_loss


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()

    def __call__(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        return jsd_loss(positive_scores, negative_scores)
