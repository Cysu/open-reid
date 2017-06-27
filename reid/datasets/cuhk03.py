from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class CUHK03(Dataset):
    url = 'https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0'
    md5 = '728939e58ad9f0ff53e521857dd8fb43'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(CUHK03, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import h5py
        import hashlib
        from scipy.misc import imsave
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'cuhk03_release.zip')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'cuhk03_release')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        matdata = h5py.File(osp.join(exdir, 'cuhk-03.mat'), 'r')

        def deref(ref):
            return matdata[ref][:].T

        def dump_(refs, pid, cam, fnames):
            for ref in refs:
                img = deref(ref)
                if img.size == 0 or img.ndim < 2: break
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(fnames))
                imsave(osp.join(images_dir, fname), img)
                fnames.append(fname)

        identities = []
        for labeled, detected in zip(
                matdata['labeled'][0], matdata['detected'][0]):
            labeled, detected = deref(labeled), deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                images = [[], []]
                dump_(labeled[i, :5], pid, 0, images[0])
                dump_(detected[i, :5], pid, 0, images[0])
                dump_(labeled[i, 5:], pid, 1, images[1])
                dump_(detected[i, 5:], pid, 1, images[1])
                identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'cuhk03', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save training and test splits
        splits = []
        view_counts = [deref(ref).shape[0] for ref in matdata['labeled'][0]]
        vid_offsets = np.r_[0, np.cumsum(view_counts)]
        for ref in matdata['testsets'][0]:
            test_info = deref(ref).astype(np.int32)
            test_pids = sorted(
                [int(vid_offsets[i-1] + j - 1) for i, j in test_info])
            trainval_pids = list(set(range(vid_offsets[-1])) - set(test_pids))
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
