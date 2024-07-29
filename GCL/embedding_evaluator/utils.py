import numpy as np
import torch
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score


def train_test_split(num_samples: int, train_size: float = 0.1, test_size: float = 0.8):
    assert train_size + test_size < 1
    train_size = int(num_samples * train_size)
    test_size = int(num_samples * test_size)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test':  indices[test_size + train_size:]
    }


def score(y_pred: np.array, y_test: np.array, measure: str) -> float:
    if measure == 'accuracy':
        return accuracy_score(y_test, y_pred)
    elif measure == 'precision':
        return precision_score(y_test, y_pred)
    elif measure == 'recall':
        return recall_score(y_test, y_pred)
    elif measure == 'micro_f1':
        return fbeta_score(y_test, y_pred, beta=1, average='micro')
    elif measure == 'macro_f1':
        return fbeta_score(y_test, y_pred, beta=1, average='macro')
    elif measure == 'auc':
        return roc_auc_score(y_test, y_pred)
