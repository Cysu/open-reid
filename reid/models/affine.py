import torch
from torch import nn
from torch.nn import Parameter


def affine(x, weight, bias=None, axis=1):
    shape = [1] * x.dim()
    shape[axis] = -1
    x = x * weight.view(*shape).expand_as(x)
    if bias is not None:
        x = x + bias.view(*shape).expand_as(x)
    return x


class Affine(nn.Module):
    def __init__(self, num_features, axis=1, bias=True):
        super(Affine, self).__init__()
        self.num_features = num_features
        self.axis = axis

        self.weight = Parameter(torch.Tensor(num_features))
        if bias:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x):
        self._check_input_dim(x)
        return affine(x, self.weight, bias=self.bias, axis=self.axis)

    def reset_parameters(self):
        self.weight.data.fill_(1)
        if self.bias is not None:
            self.bias.data.zero_()

    def _check_input_dim(self, x):
        if x.dim() <= self.axis:
            raise ValueError('got {}-dim tensor, expected at least {}-dim'
                             .format(x.dim(), self.axis + 1))
        if x.size(self.axis) != self.num_features:
            raise ValueError('got {}-feature tensor along axis {}, expected {}'
                             .format(x.size(self.axis), self.axis,
                                     self.num_features))

    def __repr__(self):
        return ('{name}{num_features}, axis={axis}, bias={bias}'
                .format(name=self.__class__.__name__,
                        num_features=self.num_features,
                        axis=self.axis,
                        bias=self.bias is not None))
