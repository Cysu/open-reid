from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class CUHK01(Dataset):
    url = 'https://docs.google.com/spreadsheet/viewform?formkey=dF9pZ1BFZkNiMG1oZUdtTjZPalR0MGc6MA'
    md5 = 'e6d55c0da26d80cda210a2edeb448e98'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(CUHK01, self).__init__(root, split_id=split_id)

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

        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'CUHK01.zip')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'campus')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        identities = [[[] for _ in range(2)] for _ in range(971)]

        files = sorted(glob(osp.join(exdir, '*.png')))
        for fpath in files:
            fname = osp.basename(fpath)
            pid, cam = int(fname[:4]), int(fname[4:7])
            assert 1 <= pid <= 971
            assert 1 <= cam <= 4
            pid, cam = pid - 1, (cam - 1) // 2
            fname = ('{:08d}_{:02d}_{:04d}.png'
                     .format(pid, cam, len(identities[pid][cam])))
            identities[pid][cam].append(fname)
            shutil.copy(fpath, osp.join(images_dir, fname))

        # Save meta information into a json file
        meta = {'name': 'cuhk01', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
