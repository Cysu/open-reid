from __future__ import print_function
import time

import torch
from torch.autograd import Variable

from reid.evaluation import accuracy, cmc
from reid.utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, model, criterion, args):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.args = args

    def train(self, epoch, data_loader, optimizer):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        end = time.time()
        for i, (imgs, _, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            pids = pids.cuda()
            inputs = Variable(imgs)
            targets = Variable(pids)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            prec1, = accuracy(outputs.data, pids)
            losses.update(loss.data[0], imgs.size(0))
            top1.update(prec1[0], imgs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Top1 {:.2%} ({:.2%})\t'.format(
                    epoch, i, len(data_loader),
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.avg,
                    losses.val, losses.avg, top1.val, top1.avg))


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


class Evaluator(object):
    def __init__(self, model, args):
        super(Evaluator, self).__init__()
        self.model = model
        self.args = args

    def evaluate(self, data_loader, query, gallery):
        features = self.extract_features(data_loader)
        distmat = pairwise_distance(features, query, gallery)
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
        cmc_scores = cmc(distmat, query_ids, gallery_ids,
                         query_cams, gallery_cams)
        print('CMC Scores:')
        for k in [1, 5, 10]:
            print('  top-{:<3}{:6.1%}'.format(k, cmc_scores[k-1]))
        return cmc_scores[0]

    def extract_features(self, data_loader):
        self.model.eval()

        features = {}

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        for i, (imgs, fnames, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs = Variable(imgs, volatile=True)
            outputs = self.model(inputs).data

            assert len(fnames) == outputs.size(0)
            for fname, output in zip(fnames, outputs):
                features[fname] = output

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Evaluate: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'.format(
                    i, len(data_loader),
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.avg))

        return features
