from unittest import TestCase

import torch
from torch.autograd import Variable

from reid.models.affine import Affine


class TestAffine(TestCase):
    def test_all(self):
        n, d = 10, 8
        layer = Affine(d, axis=1, bias=True)
        w = torch.range(0, d - 1)
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(w)
        x = Variable(torch.rand(n, d), requires_grad=True)
        y = layer(x)
        w = w.view(1, d).expand_as(x)
        self.assertTrue(torch.equal(y.data, x.data * w + w))
