from unittest import TestCase
import numpy as np

from reid.evaluation_metrics import cmc


class TestCMC(TestCase):
    def test_only_distmat(self):
        distmat = np.array([[0, 1, 2, 3, 4],
                            [1, 0, 2, 3, 4],
                            [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4],
                            [1, 2, 3, 4, 0]])
        ret = cmc(distmat)
        self.assertTrue(np.all(ret[:5] == [0.6, 0.6, 0.8, 1.0, 1.0]))

    def test_duplicate_ids(self):
        distmat = np.tile(np.arange(4), (4, 1))
        query_ids = [0, 0, 1, 1]
        gallery_ids = [0, 0, 1, 1]
        ret = cmc(distmat, query_ids=query_ids, gallery_ids=gallery_ids, topk=4,
                  separate_camera_set=False, single_gallery_shot=False)
        self.assertTrue(np.all(ret == [0.5, 0.5, 1, 1]))

    def test_duplicate_cams(self):
        distmat = np.tile(np.arange(5), (5, 1))
        query_ids = [0,0,0,1,1]
        gallery_ids = [0,0,0,1,1]
        query_cams = [0,0,0,0,0]
        gallery_cams = [0,1,1,1,1]
        ret = cmc(distmat, query_ids=query_ids, gallery_ids=gallery_ids,
                  query_cams=query_cams, gallery_cams=gallery_cams, topk=5,
                  separate_camera_set=False, single_gallery_shot=False)
        self.assertTrue(np.all(ret == [0.6, 0.6, 0.6, 1, 1]))
