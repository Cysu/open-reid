from unittest import TestCase


class TestVIPeR(TestCase):
    def test_init(self):
        import os.path as osp
        from reid.datasets.viper import VIPeR
        from reid.utils.serialization import read_json

        root, split_id, num_val = '/tmp/open-reid/viper', 0, 100
        dataset = VIPeR(root, split_id=split_id, num_val=num_val, download=True)

        self.assertTrue(osp.isfile(osp.join(root, 'meta.json')))
        self.assertTrue(osp.isfile(osp.join(root, 'splits.json')))
        meta = read_json(osp.join(root, 'meta.json'))
        self.assertEquals(len(meta['identities']), 632)
        splits = read_json(osp.join(root, 'splits.json'))
        self.assertEquals(len(splits), 10)

        self.assertDictEqual(meta, dataset.meta)
        self.assertDictEqual(splits[split_id], dataset.split)
