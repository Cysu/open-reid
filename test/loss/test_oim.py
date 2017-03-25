from unittest import TestCase


class TestOIMLoss(TestCase):
    def test_forward_backward(self):
        import torch
        import torch.nn.functional as F
        from torch.autograd import Variable
        from reid.loss import OIMLoss
        criterion = OIMLoss(3, 3, scalar=1.0, size_average=False)
        criterion.lut = torch.eye(3)
        x = Variable(torch.randn(3, 3), requires_grad=True)
        y = Variable(torch.range(0, 2).long())
        loss = criterion(x, y)
        loss.backward()
        probs = F.softmax(x)
        grads = probs.data - torch.eye(3)
        abs_diff = torch.abs(grads - x.grad.data)
        self.assertEquals(torch.log(probs).diag().sum(), -loss)
        self.assertTrue(torch.max(abs_diff) < 1e-6)
