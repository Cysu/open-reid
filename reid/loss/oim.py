import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):
    def __init__(self, lut):
        super(OIM, self).__init__()
        self.lut = lut

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = 0.5 * (self.lut[y] + x)
        return grad_inputs, None


def oim(inputs, targets, lut):
    return OIM(lut)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average)
        return loss, inputs
