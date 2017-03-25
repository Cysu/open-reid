import torch

from .evaluators import extract_features
from .metric_learning import get_metric


class DistanceMetric(object):
    def __init__(self, algorithm='euclidean', *args, **kwargs):
        super(DistanceMetric, self).__init__()
        self.algorithm = algorithm
        self.metric = get_metric(algorithm, *args, **kwargs)

    def train(self, model, data_loader):
        if self.algorithm == 'euclidean': return
        features, labels = extract_features(model, data_loader)
        features = torch.stack(features.values()).numpy()
        labels = torch.cat(labels.values()).numpy()
        self.metric.fit(features, labels)
