from __future__ import print_function, absolute_import
import os
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class DukeMTMC(Dataset):
    url = 'https://drive.google.com/open?id=0B0VOCNYh8HeRSDRwczZIT0lZTG8'
    md5 = '286aaef9ba5db58853d91b66a028923b'

    def __init__(self, root, split_id=0, num_val=0.3, download=False):
        super(DukeMTMC, self).__init__(root, split_id=split_id)

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

        import re
        import hashlib
        import shutil
        import tarfile
        from glob import glob

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'Duke.tar.gz')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'Duke')
        if not osp.isdir(exdir):
            mkdir_if_missing(exdir)
            print("Extracting tar file")
            cwd = os.getcwd()
            tar = tarfile.open(fpath, 'r:gz')
            os.chdir(exdir)
            tar.extractall()
            tar.close()
            os.chdir(cwd)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        identities = []
        all_pids = {}

        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                assert 1 <= cam <= 8
                cam -= 1
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                pids.add(pid)
                if pid >= len(identities):
                    assert pid == len(identities)
                    identities.append([[] for _ in range(8)])  # 8 camera views
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        trainval_pids = register('bounding_box_train')
        gallery_pids = register('bounding_box_test')
        query_pids = register('query')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'DukeMTMC', 'shot': 'multiple', 'num_cameras': 8,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
