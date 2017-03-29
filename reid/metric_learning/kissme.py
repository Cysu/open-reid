from __future__ import absolute_import

import numpy as np
from metric_learn.base_metric import BaseMetricLearner


def validate_cov_matrix(M, threshold=1e-10, eps=1e-6):
    try:
        _ = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eig(M)
        w[w <= threshold] = eps
        M = (v * w).dot(v.T)
    return M


class KISSME(BaseMetricLearner):
    def __init__(self):
        self.M_ = None

    def metric(self):
        return self.M_

    def fit(self, X, y=None):
        n = X.shape[0]
        if y is None:
            y = np.arange(n)
        X1, X2 = np.meshgrid(np.arange(n), np.arange(n))
        X1, X2 = X1[X1 < X2], X2[X1 < X2]
        matches = (y[X1] == y[X2])
        num_matches = matches.sum()
        num_non_matches = len(matches) - num_matches
        idxa = X1[matches]
        idxb = X2[matches]
        S = X[idxa] - X[idxb]
        C1 = S.transpose().dot(S) / num_matches
        p = np.random.choice(num_non_matches, num_matches, replace=False)
        idxa = X1[~matches]
        idxb = X2[~matches]
        idxa = idxa[p]
        idxb = idxb[p]
        S = X[idxa] - X[idxb]
        C0 = S.transpose().dot(S) / num_matches
        self.M_ = np.linalg.inv(C1) - np.linalg.inv(C0)
        self.M_ = validate_cov_matrix(self.M_)
        self.X_ = X
