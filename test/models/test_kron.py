from unittest import TestCase

import torch
from torch.autograd import Variable

from reid.models.kron import KronMatching


class TestKronMatching(TestCase):
    def test_all(self):
        N, C, H, W = 2, 3, 5, 5
        x = Variable(torch.rand(N, C, H, W).cuda(), requires_grad=True)
        y = Variable(torch.rand(N, C, H, W).cuda(), requires_grad=True)

        layer = KronMatching().cuda()
        z = layer(x, y)

        # Forward and backward check
        t = torch.zeros(N, (2*H-1)*(2*W-1), H, W).cuda()
        dx = torch.zeros(N, C, H, W).cuda()
        dy = torch.zeros(N, C, H, W).cuda()
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    for p in range(H):
                        for q in range(W):
                            value = x.data[n,:,i,j].dot(y.data[n,:,p,q])
                            index = (p-i+H-1) * (2*W-1) + (q-j+W-1)
                            t[n, index, i, j] = value
                            dx[n, :, i, j] += y.data[n, :, p, q]
                            dy[n, :, p, q] += x.data[n, :, i, j]

        loss = z.sum()
        loss.backward()

        eps = x.data.abs().mean() * 1e-5
        self.assertLess(torch.abs(z.data - t).max(), eps)
        self.assertLess(torch.abs(x.grad.data - dx).max(), eps)
        self.assertLess(torch.abs(y.grad.data - dy).max(), eps)
