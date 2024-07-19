from abc import ABC

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from torch import nn
from tqdm import tqdm

from GCL.embedding_evaluator.utils import score, train_test_split


class Evaluator(ABC):
    def __init__(self):
        pass

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, masks: dict) -> dict:
        for key in ['train', 'test']:
            assert key in masks
        pass

    def __call__(self, x: torch.Tensor, y: torch.Tensor, masks: dict) -> dict:
        for key in ['train', 'test']:
            assert key in masks

        return self.evaluate(x, y, masks)


class LogisticRegression(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(Evaluator):
    def __init__(self, epochs: int = 5000, learning_rate: float = 0.001, measures=None):
        super().__init__()
        if measures is None:
            measures = ['accuracy']
        self.measures = measures
        self.epochs = epochs
        self.learning_rate = learning_rate

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, masks: dict) -> tuple:
        device = x.device

        num_features = x.shape[1]
        num_classes = len(torch.unique(y))
        train_x, test_x, valid_x = x[masks['train']], x[masks['test']], x[masks['valid']]
        train_y, test_y, valid_y = y[masks['train']], y[masks['test']], y[masks['valid']]

        performances = {measure: [] for measure in self.measures}

        model = LogisticRegression(num_features, num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs // 5, gamma=0.1)

        # train model
        with tqdm(total=self.epochs, desc='Evaluating embedding by Logistic Regression') as pbar:
            epoch_performance = {}
            for epoch in range(self.epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_x)
                train_loss = loss_fn(outputs, train_y)

                train_loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % max((self.epochs // 100), 20) == 0:
                    model.eval()
                    y_test = test_y.detach().cpu().numpy()
                    y_pred = model(test_x).detach().cpu().numpy().argmax(axis=1)
                    for measure in self.measures:
                        s = score(y_pred, y_test, measure)
                        performances[measure].append(s)
                        epoch_performance[measure] = s

                pbar.set_postfix(epoch_performance)
                pbar.update(1)

        return performances, [np.max(performances[measure]) for measure in self.measures]


class SVCEvaluator(Evaluator):
    def __init__(self, measures: list = None, params=None):
        super().__init__()
        if measures is None:
            measures = ['accuracy']

        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        self.params = params
        self.svc = SVC()
        self.measures = measures

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, masks: dict) -> dict:
        train_x, test_x, valid_x = x[masks['train']], x[masks['test']], x[masks['valid']]
        train_y, test_y, valid_y = y[masks['train']], y[masks['test']], y[masks['valid']]

        model = GridSearchCV(self.svc, self.params, cv=5, scoring='accuracy', verbose=0)
        model.fit(train_x.cpu().numpy(), train_y.cpu().numpy())

        performances = {measure: [] for measure in self.measures}
        for measure in self.measures:
            y_test = test_y.detach().cpu().numpy()
            y_pred = model.predict(test_x.cpu().numpy())
            s = score(y_pred, y_test, measure)
            performances[measure].append(s)

        return performances


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集大小
    num_samples = 1000
    num_features = 2

    X = torch.randn(num_samples, num_features).to(device)

    y = (torch.sum(X, dim=1) > 0).long().to(device)
    masks = train_test_split(X.size(0))
    evaluator = LREvaluator(measures=['accuracy', 'macro_f1', 'precision', 'recall'])
    performance, best_performance = evaluator(X, y, masks)
    print(best_performance)
