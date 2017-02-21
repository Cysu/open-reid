from __future__ import print_function
import time

from torch.autograd import Variable

from reid.evaluation import accuracy
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
        for i, (imgs, pids, camids) in enumerate(data_loader):
            data_time.update(time.time() - end)
            # target = target.cuda(async=True)
            input_var = Variable(imgs)
            target_var = Variable(pids)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            prec1, = accuracy(output.data, pids)
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
