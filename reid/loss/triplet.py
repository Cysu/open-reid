import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=0).sqrt()
        # Enumerate triplets (a, n, p), and save (an, ap) distance indices
        d, y = dist.data, targets.data
        indices = []
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if y[i] != y[j]: continue
                for k in range(n):
                    if y[k] == y[i]: continue
                    if d[i][k] - d[i][j] >= self.margin: continue
                    indices.append((i * n + k, i * n + j))
        indices = torch.Tensor(indices).long()
        if dist.is_cuda:
            indices = indices.cuda()
        # Since torch does not support 2d indexing yet, we need to flatten it
        dist = dist.view(-1)
        dist_an, dist_ap = dist[indices[:, 0]], dist[indices[:, 1]]
        # Compute ranking hinge loss
        y = torch.ones(dist_an.size(0))
        if dist.is_cuda:
            y = y.cuda()
        y = Variable(y)
        return self.ranking_loss(dist_an, dist_ap, y)
