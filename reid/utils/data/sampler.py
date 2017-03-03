from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import *


class PairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1, exhaustive=False):
        super(PairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        self.exhaustive = exhaustive
        # Build up positive sets and negative lists
        if not exhaustive:
            all_samples = set(np.arange(self.num_samples))
            self.pos_samples = defaultdict(set)
            for index, (_, pid, _) in enumerate(data_source):
                self.pos_samples[pid].add(index)
            self.neg_samples = {}
            for pid, samples in self.pos_samples.items():
                self.neg_samples[pid] = list(all_samples - samples)

    def __iter__(self):
        if self.exhaustive:
            for i in range(self.num_samples):
                for j in range(i + 1, self.num_samples):
                    yield i, j
            return

        indices = np.random.permutation(self.num_samples)
        for anchor_index in indices:
            _, anchor_pid, _ = self.data_source[anchor_index]
            pos_samples = self.pos_samples[anchor_pid]
            # Choose one positive sample randomly
            pos_index = np.random.choice(list(pos_samples - {anchor_index}))
            yield anchor_index, pos_index
            # Choose several negative samples randomly
            neg_indices = np.random.choice(self.neg_samples[anchor_pid],
                                           self.neg_pos_ratio, replace=False)
            for neg_index in neg_indices:
                yield anchor_index, neg_index

    def __len__(self):
        if self.exhaustive:
            return self.num_samples * (self.num_samples - 1) // 2
        return self.num_samples * (1 + self.neg_pos_ratio)
