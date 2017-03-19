from unittest import TestCase

import numpy as np
import torch
from torch.utils.data import DataLoader

from reid.datasets import VIPeR
from reid.features import FeatureDatabase, extract_features
from reid.models import InceptionNet
from reid.utils import to_numpy
from reid.utils.data import transforms, Preprocessor


class TestExtractFeatures(TestCase):
    def test_all(self):
        root, split_id, num_val = '/tmp/open-reid/viper', 0, 100
        dataset = VIPeR(root, split_id=split_id, num_val=num_val, download=True)
        data_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir,
                         transform=transforms.Compose([
                             transforms.RectScale(144, 56),
                             transforms.ToTensor(),
                         ])),
            batch_size=128, num_workers=2,
            shuffle=False, pin_memory=False)

        model = InceptionNet(num_features=256, norm=True, dropout=0.5)
        model = torch.nn.DataParallel(model).cuda()

        mem_features = extract_features(model, data_loader)

        out_file = '/tmp/open-reid/test_extract_features.h5'
        extract_features(model, data_loader, output_file=out_file)

        with FeatureDatabase(out_file, 'r') as db:
            self.assertEquals(len(mem_features), len(db))
            for name in mem_features:
                self.assertTrue(name in db)
                x, y = to_numpy(mem_features[name]), db[name]
                self.assertEquals(x.shape, y.shape)
                self.assertTrue(np.allclose(x, y))
