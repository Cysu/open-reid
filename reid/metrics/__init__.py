import torch


def pairwise_distance(features, query, gallery):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist.cpu().numpy()