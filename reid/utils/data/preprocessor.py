import os.path as osp

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        if self.root is not None:
            fname = osp.join(self.root, fname)
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid
