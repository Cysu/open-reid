from unittest import TestCase

from reid.datasets import VIPeR
from reid.utils.data.sampler import RandomPairSampler, ExhaustiveSampler


class TestPairSampler(TestCase):
    def test_exhaustive(self):
        dataset = VIPeR('/tmp/open-reid/viper',
                        split_id=0, num_val=100, download=True)
        sampler = ExhaustiveSampler(dataset.train)

        n = len(dataset.train)
        self.assertEquals(len(sampler), n * n)

        sampler_iter = iter(sampler)
        for i in range(n):
            for j in range(n):
                p, q = next(sampler_iter)
                self.assertEquals((i, j), (p, q))

    def test_random(self):
        dataset = VIPeR('/tmp/open-reid/viper',
                        split_id=0, num_val=100, download=True)
        sampler = RandomPairSampler(dataset.train, neg_pos_ratio=1)

        n = len(dataset.train)
        self.assertEquals(len(sampler), n * 2)

        sampler_iter = iter(sampler)
        for _ in range(n):
            i, j = next(sampler_iter)
            self.assertEquals(dataset.train[i][1], dataset.train[j][1])
            self.assertNotEquals(dataset.train[i][0], dataset.train[j][0])
            i, j = next(sampler_iter)
            self.assertNotEquals(dataset.train[i][1], dataset.train[j][1])
