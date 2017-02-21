import os.path as osp

import numpy as np

from reid.utils.serialization import read_json


def _pluck(identities, indices):
    ret = []
    for pid in indices:
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                ret.append((fname, pid, camid))
    return ret


class Dataset(object):
    def __init__(self):
        self.root = None
        self.split_id = None
        self.meta = None
        self.split = None
        self.training, self.validation = [], []
        self.test_query, self.test_gallery = [], []

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_val=0.3):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.training = _pluck(identities, train_pids)
        self.validation = _pluck(identities, val_pids)
        self.test_query = _pluck(identities, self.split['test_query'])
        self.test_gallery = _pluck(identities, self.split['test_gallery'])

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
