from __future__ import print_function
import time

from torch.autograd import Variable

from .evaluation import accuracy
from .utils.meters import AverageMeter


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
        for i, pair in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs1, inputs2, targets = self._parse_data(pair)

            outputs = self.model(inputs1, inputs2)
            loss = self.criterion(outputs, targets)

            prec1, = accuracy(outputs.data, targets.data)
            losses.update(loss.data[0], inputs1.size(0))
            top1.update(prec1[0], inputs1.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Top1 {:.2%} ({:.2%})\t'.format(
                    epoch, i + 1, len(data_loader),
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.avg,
                    losses.val, losses.avg, top1.val, top1.avg))

    def _parse_data(self, pair):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = pair
        inputs1, inputs2 = Variable(imgs1), Variable(imgs2)
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs1, inputs2, targets
