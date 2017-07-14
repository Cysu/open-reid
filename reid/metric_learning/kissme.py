from __future__ import absolute_import

import numpy as np
from metric_learn.base_metric import BaseMetricLearner


def validate_cov_matrix(M):
    M = (M + M.T) * 0.5
    k = 0
    I = np.eye(M.shape[0])
    while True:
        try:
            _ = np.linalg.cholesky(M)
            break
        except np.linalg.LinAlgError:
            # Find the nearest positive definite matrix for M. Modified from
            # http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            # Might take several minutes
            k += 1
            w, v = np.linalg.eig(M)
            min_eig = v.min()
            M += (-min_eig * k * k + np.spacing(min_eig)) * I
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
