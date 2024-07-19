import numpy as np
import torch
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score


def train_test_split(num_samples: int, test_size: float = 0.2, valid_size: float = 0.0) -> dict:
    train_size = 1.0 - test_size - valid_size
    indices = torch.randperm(num_samples)

    num_train = int(train_size * num_samples)
    num_test = int(test_size * num_samples)

    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    test_mask[indices[num_train:num_train + num_test]] = True
    val_mask[indices[num_train + num_test:]] = True

    return {'train': train_mask, 'test': test_mask, 'valid': val_mask}


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
