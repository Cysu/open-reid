from torch.utils.data import DataLoader

from .evaluation.routines import evaluate_cmc
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
        return evaluate_cmc(distmat, query, gallery)


class SiameseEvaluator(object):
    def __init__(self, base_model, embed_model, dist_fn=None):
        super(SiameseEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.dist_fn = dist_fn

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

        if cache_file is not None:
            features.close()

        # Convert embeddings to distance matrix
        if self.dist_fn is not None:
            embeddings = self.dist_fn(embeddings)
        distmat = embeddings.contiguous().view(len(query), len(gallery))

        # Evaluate CMC scores
        return evaluate_cmc(distmat, query, gallery)
