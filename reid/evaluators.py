import numpy as np
from torch.utils.data import DataLoader

from .evaluation.routines import evaluate_all
from .features import FeatureDatabase, extract_features, extract_embeddings
from .metrics import pairwise_distance
from .utils.data.preprocessor import KeyValuePreprocessor
from .utils.data.sampler import ExhaustiveSampler


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery):
        features = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)


class SiameseEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        super(SiameseEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn

    def evaluate(self, data_loader, query, gallery, cache_file=None):
        # Extract features image by image
        features = extract_features(self.base_model, data_loader,
                                    output_file=cache_file)
        if cache_file is not None:
            features = FeatureDatabase(cache_file, 'r')

        # Build a data loader for exhaustive (query, gallery) pairs
        query_keys = [fname for fname, _, _ in query]
        gallery_keys = [fname for fname, _, _ in gallery]
        data_loader = DataLoader(
            KeyValuePreprocessor(features),
            sampler=ExhaustiveSampler(query_keys, gallery_keys,
                                      return_index=False),
            batch_size=min(len(gallery), 4096),
            num_workers=1, pin_memory=False)

        # Extract embeddings of each (query, gallery) pair
        embeddings = extract_embeddings(self.embed_model, data_loader)
        if self.embed_dist_fn is not None:
            embeddings = self.embed_dist_fn(embeddings)

        if cache_file is not None:
            features.close()

        # Convert embeddings to distance matrix
        distmat = embeddings.contiguous().view(len(query), len(gallery))

        # Evaluate CMC scores
        return evaluate_all(distmat, query=query, gallery=gallery)


class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn

    def evaluate(self, data_loader, query, gallery, cache_file=None,
                 rerank_topk=20):
        # Extract features image by image
        features = extract_features(self.base_model, data_loader,
                                    output_file=cache_file)
        if cache_file is not None:
            features = FeatureDatabase(cache_file, 'r')

        # Compute pairwise distance and evaluate for the first stage
        distmat = pairwise_distance(features, query, gallery)
        print("First stage evaluation:")
        evaluate_all(distmat, query=query, gallery=gallery)

        # Sort according to the first stage distance
        distmat = distmat.cpu().numpy()
        rank_indices = np.argsort(distmat, axis=1)

        # Build a data loader for topk predictions for each query
        pair_samples = []
        for i, indices in enumerate(rank_indices):
            query_fname, _, _ = query[i]
            for j in indices[:rerank_topk]:
                gallery_fname, _, _ = gallery[j]
                pair_samples.append((query_fname, gallery_fname))

        data_loader = DataLoader(
            KeyValuePreprocessor(features),
            sampler=pair_samples,
            batch_size=min(len(gallery), 4096),
            num_workers=1, pin_memory=False)

        # Extract embeddings of each pair
        embeddings = extract_embeddings(self.embed_model, data_loader)
        if self.embed_dist_fn is not None:
            embeddings = self.embed_dist_fn(embeddings)

        if cache_file is not None:
            features.close()

        # Merge two-stage distances
        for k, embed in enumerate(embeddings):
            i, j = k // rerank_topk, k % rerank_topk
            distmat[i, rank_indices[i, j]] = embed
        for i, indices in enumerate(rank_indices):
            bar = max(distmat[i][indices[:rerank_topk]])
            gap = max(bar + 1. - distmat[i, indices[rerank_topk]], 0)
            if gap > 0:
                distmat[i][indices[rerank_topk:]] += gap

        print("Second stage evaluation:")
        return evaluate_all(distmat, query, gallery)
