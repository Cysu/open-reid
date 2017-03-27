from unittest import TestCase


class TestCUHK01(TestCase):
    def test_init(self):
        import os.path as osp
        from reid.datasets import CUHK01
        from reid.utils.serialization import read_json

        root, split_id, num_val = '/tmp/open-reid/cuhk01', 0, 100
        dataset = CUHK01(root, split_id=split_id, num_val=num_val, download=True)

        self.assertTrue(osp.isfile(osp.join(root, 'meta.json')))
        self.assertTrue(osp.isfile(osp.join(root, 'splits.json')))
        meta = read_json(osp.join(root, 'meta.json'))
        self.assertEquals(len(meta['identities']), 971)
        splits = read_json(osp.join(root, 'splits.json'))
        self.assertEquals(len(splits), 10)

        self.assertDictEqual(meta, dataset.meta)
        self.assertDictEqual(splits[split_id], dataset.split)
