from unittest import TestCase


class TestVIPeR(TestCase):
    def test(self):
        import os.path as osp
        from reid.datasets.viper import VIPeR
        from reid.utils.serialization import read_json
        root = '/tmp/open-reid/viper'
        dataset = VIPeR(root, download=True)
        self.assertTrue(osp.isfile(osp.join(root, 'meta.json')))
        self.assertTrue(osp.isfile(osp.join(root, 'splits.json')))
        meta = read_json(osp.join(root, 'meta.json'))
        self.assertEquals(len(meta['identities']), 632)
        splits = read_json(osp.join(root, 'splits.json'))
        self.assertEquals(len(splits), 10)
