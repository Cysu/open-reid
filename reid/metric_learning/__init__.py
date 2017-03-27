import numpy as np
from metric_learn import *

from .euclidean import Euclidean
from .kissme import KISSME

__factory = {
    'euclidean': Euclidean,
    'kissme': KISSME,
    'itml': ITML_Supervised,
    'lmnn': LMNN,
    'lsml': LSML_Supervised,
    'sdml': SDML_Supervised,
    'nca': NCA,
    'lfda': LFDA,
    'rca': RCA_Supervised,
}


def get_metric(algorithm, *args, **kwargs):
    if algorithm not in __factory:
        raise KeyError("Unknown metric:", algorithm)
    return __factory[algorithm](*args, **kwargs)


def validate_cov_matrix(M, threshold=1e-10, eps=1e-6):
    try:
        _ = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eig(M)
        w[w <= threshold] = eps
        M = (v * w).dot(v.T)
    return M
