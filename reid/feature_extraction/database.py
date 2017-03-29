from __future__ import absolute_import

import h5py
import numpy as np
from torch.utils.data import Dataset


class FeatureDatabase(Dataset):
    def __init__(self, *args, **kwargs):
        super(FeatureDatabase, self).__init__()
        self.fid = h5py.File(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, keys):
        if isinstance(keys, (tuple, list)):
            return [self._get_single_item(k) for k in keys]
        return self._get_single_item(keys)

    def _get_single_item(self, key):
        return np.asarray(self.fid[key])

    def __setitem__(self, key, value):
        if key in self.fid:
            if self.fid[key].shape == value.shape and \
               self.fid[key].dtype == value.dtype:
                self.fid[key][...] = value
            else:
                del self.fid[key]
                self.fid.create_dataset(key, data=value)
        else:
            self.fid.create_dataset(key, data=value)

    def __delitem__(self, key):
        del self.fid[key]

    def __len__(self):
        return len(self.fid)

    def __iter__(self):
        return iter(self.fid)

    def flush(self):
        self.fid.flush()

    def close(self):
        self.fid.close()
