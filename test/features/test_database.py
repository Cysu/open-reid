from unittest import TestCase

import numpy as np

from reid.feature_extraction.database import FeatureDatabase


class TestFeatureDatabase(TestCase):
    def test_all(self):
        with FeatureDatabase('/tmp/open-reid/test.h5', 'w') as db:
            db['img1'] = np.random.rand(3, 8, 8).astype(np.float32)
            db['img2'] = np.arange(10)
            db['img2'] = np.arange(10).reshape(2, 5).astype(np.float32)
        with FeatureDatabase('/tmp/open-reid/test.h5', 'r') as db:
            self.assertTrue('img1' in db)
            self.assertTrue('img2' in db)
            self.assertEquals(db['img1'].shape, (3, 8, 8))
            x = db['img2']
            self.assertEquals(x.shape, (2, 5))
            self.assertTrue(np.all(x == np.arange(10).reshape(2, 5)))

