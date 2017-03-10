import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def kron_matching(*inputs):
    assert len(inputs) == 2
    assert inputs[0].dim() == 4 and inputs[1].dim() == 4
    assert inputs[0].size() == inputs[1].size()
    N, C, H, W = inputs[0].size()

    # Convolve every feature vector from inputs[0] with inputs[1]
    #   In: x0, x1 = N x C x H x W
    #   Proc: weight = x0, permute to (NxHxW) x C x 1 x 1
    #         input = x1, view as 1 x (NxC) x H x W
    #   Out: out = F.conv2d(input, weight, groups=N)
    #            = 1 x (NxHxW) x H x W, view as N x H x W x (HxW)
    w = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)
    x = inputs[1].view(1, N * C, H, W)
    x = F.conv2d(x, w, groups=N)
    x = x.view(N, H, W, H, W)

    # Generate the index used for scatter
    #    index = H x W x H x W
    #    index[i,j,p,q] = (p-i+H-1) * (2*W-1) + (q-j+W-1)
    i1 = torch.range(0, H - 1).long()
    i1 = i1.expand(H, H) - i1.unsqueeze(1).expand(H, H)
    i1 = (i1 + H - 1) * (2 * W - 1)
    i2 = torch.range(0, W - 1).long()
    i2 = i2.expand(W, W) - i2.unsqueeze(1).expand(W, W)
    i2 = i2 + W - 1
    i1 = i1.view(H, 1, H, 1).expand(H, W, H, W)
    i2 = i2.view(1, W, 1, W).expand(H, W, H, W)
    index = i1 + i2

    # Scatter to the output
    #   In: x = N x H x W x H x W, index = H x W x H x W
    #   Proc: out = N x H x W x (2H-1)(2W-1)
    #         where out[n, i, j, index[n,i,j,k]] = x[n, i, j, k]
    #   Out: permute out to N x (2H-1)(2W-1) x H x W
    x = x.view(N, H, W, H * W)
    index = index.view(1, H, W, H * W).expand_as(x)
    index = Variable(index, requires_grad=False)
    out = Variable(torch.zeros(N, H, W, (2 * H - 1) * (2 * W - 1)))
    out.scatter_(3, index, x)
    out = out.permute(0, 3, 1, 2).contiguous()

    return out


class KronMatching(nn.Module):
    def __init__(self):
        super(KronMatching, self).__init__()

    def forward(self, *inputs):
        return kron_matching(*inputs)
