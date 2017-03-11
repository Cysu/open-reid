import numpy as np

from .features import extract_features
from .metrics import pairwise_distance


def mine_hard_pairs(model, data_loader, margin=0):
    model.eval()
    # Compute pairwise distance
    features = extract_features(model, data_loader, print_freq=1)
    distmat = pairwise_distance(features)
    distmat = distmat.cpu().numpy()
    # Get the pids
    dataset = data_loader.dataset.dataset
    pids = np.asarray([pid for _, pid, _ in dataset])
    # Find the hard triplets
    pairs = []
    for i, d in enumerate(distmat):
        pos_indices = np.where(pids == pids[i])[0]
        threshold = max(d[pos_indices]) + margin
        neg_indices = np.where(pids != pids[i])[0]
        pairs.extend([(i, p) for p in pos_indices])
        pairs.extend([(i, n) for n in neg_indices if threshold >= d[n]])
    return pairs


def mine_hard_triplets(model, data_loader, margin=0):
    model.eval()
    # Compute pairwise distance
    features = extract_features(model, data_loader, print_freq=1)
    distmat = pairwise_distance(features)
    distmat = distmat.cpu().numpy()
    # Get the pids
    dataset = data_loader.dataset.dataset
    pids = np.asarray([pid for _, pid, _ in dataset])
    # Find the hard triplets
    triplets = []
    for i, d in enumerate(distmat):
        pos_indices = np.where(pids == pids[i])[0]
        neg_indices = np.where(pids != pids[i])[0]
        for p in pos_indices:
            for n in neg_indices:
                if d[p] + margin >= d[n]:
                    triplets.append((i, p, n))
    return triplets
