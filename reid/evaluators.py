from .evaluation.routines import evaluate_all
from .features import extract_features
from .metrics import pairwise_distance


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery):
        features = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)
