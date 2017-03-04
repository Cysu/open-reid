from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import *


def _choose_from(start, end, excluding=None, size=1, replace=False):
    num = end - start + 1
    if excluding is None:
        return np.random.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluding
    num_ex = ex_end - ex_start + 1
    num -= num_ex
    inds = np.random.choice(num, size=size, replace=replace) + start
    inds += (inds >= ex_start) * num_ex
    return inds


class PairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1, exhaustive=False):
        super(PairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        self.exhaustive = exhaustive

        if not exhaustive:
            # Sort by pid
            indices = np.argsort(np.asarray(data_source)[:, 1])
            self.index_map = dict(zip(np.arange(self.num_samples), indices))
            # Get the range of indices for each pid
            self.index_range = defaultdict(lambda: [self.num_samples, -1])
            for i, j in enumerate(indices):
                _, pid, _ = data_source[j]
                self.index_range[pid][0] = min(self.index_range[pid][0], i)
                self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        if self.exhaustive:
            for i in range(self.num_samples):
                for j in range(i + 1, self.num_samples):
                    yield i, j
            return

        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # positive sample
            start, end = self.index_range[pid]
            pos_index = _choose_from(start, end, excluding=(i, i))[0]
            yield anchor_index, self.index_map[pos_index]
            # negative samples
            neg_indices = _choose_from(0, self.num_samples - 1,
                                       excluding=(start, end),
                                       size=self.neg_pos_ratio)
            for neg_index in neg_indices:
                yield anchor_index, self.index_map[neg_index]

    def __len__(self):
        if self.exhaustive:
            return self.num_samples * (self.num_samples - 1) // 2
        return self.num_samples * (1 + self.neg_pos_ratio)
