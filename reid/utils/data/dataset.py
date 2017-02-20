import os.path as osp

from reid.utils.serialization import read_json


class Dataset(object):
    def __init__(self):
        self.root = None
        self.split_id = None
        self.meta = None
        self.split = None

    def load(self):
        self.meta = read_json(osp.join(self.root, 'meta.json'))
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("Split id exceeds the total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
