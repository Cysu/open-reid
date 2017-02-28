from unittest import TestCase


class TestMarket1501(TestCase):
    def test_init(self):
        import os.path as osp
        from reid.datasets.market1501 import Market1501
        from reid.utils.serialization import read_json

        root, split_id, num_val = '/tmp/open-reid/market1501', 0, 100
        dataset = Market1501(root, split_id=split_id, num_val=num_val,
                             download=True)

        self.assertTrue(osp.isfile(osp.join(root, 'meta.json')))
        self.assertTrue(osp.isfile(osp.join(root, 'splits.json')))
        meta = read_json(osp.join(root, 'meta.json'))
        self.assertEquals(len(meta['identities']), 1502)
        splits = read_json(osp.join(root, 'splits.json'))
        self.assertEquals(len(splits), 1)

        self.assertDictEqual(meta, dataset.meta)
        self.assertDictEqual(splits[split_id], dataset.split)
