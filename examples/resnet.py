from __future__ import print_function
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from reid.datasets import get_dataset
from reid.loss.oim import OIMLoss
from reid.loss.triplet import TripletLoss
from reid.models import ResNet
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(dataset_name, split_id, data_dir, batch_size, workers,
             num_instances):
    root = osp.join(data_dir, dataset_name)

    dataset = get_dataset(dataset_name, root,
                          split_id=split_id, num_val=100, download=True)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_processor = Preprocessor(dataset.train, root=dataset.images_dir,
                                   transform=transforms.Compose([
                                       transforms.RectScale(256, 256),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalizer,
                                   ]))
    if num_instances > 0:
        train_loader = DataLoader(
            train_processor, batch_size=batch_size, num_workers=workers,
            sampler=RandomIdentitySampler(dataset.train, num_instances))
    else:
        train_loader = DataLoader(
            train_processor, batch_size=batch_size, num_workers=workers,
            shuffle=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(256, 256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(256, 256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.loss == 'triplet':
        assert args.num_instances > 1, 'TripletLoss requires num_instances > 1'
        assert args.batch_size % args.num_instances == 0, \
            'num_instances should divide batch_size'
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.workers, args.num_instances)

    # Create model
    if args.loss == 'xentropy':
        model = ResNet(args.depth, pretrained=True,
                       num_classes=dataset.num_train_ids,
                       num_features=args.features, dropout=args.dropout)
    elif args.loss == 'oim':
        model = ResNet(args.depth, pretrained=True, num_features=args.features,
                       norm=True, dropout=args.dropout)
    elif args.loss == 'triplet':
        model = ResNet(args.depth, pretrained=True,
                       num_features=1024, num_classes=args.features,
                       dropout=args.dropout)
    else:
        raise ValueError("Cannot recognize loss type:", args.loss)
    model = torch.nn.DataParallel(model).cuda()

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> start epoch {}  best top1 {:.1%}"
              .format(args.start_epoch, best_top1))
    else:
        best_top1 = 0

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    if args.loss == 'xentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'oim':
        criterion = OIMLoss(model.module.num_features, dataset.num_train_ids,
                            scalar=args.oim_scalar, momentum=args.oim_momentum)
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.triplet_margin)
    else:
        raise ValueError("Cannot recognize loss type:", args.loss)
    criterion.cuda()

    # Optimizer: different learning rates for pretrained and new layers
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups,
                                    lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("Cannot recognize optimizer type:", args.optimizer)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        if args.optimizer == 'sgd':
            lr = args.lr * (0.1 ** (epoch // 40))
        elif args.optimizer == 'adam':
            lr = args.lr if epoch <= 400 else \
                args.lr * (0.001 ** (epoch - 400) / 275)
        else:
            raise ValueError("Cannot recognize optimizer type:", args.optimizer)
        for g in optimizer.param_groups:
            g['lr'] = lr * g['lr_mult']

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        if args.loss == 'oim':
            criterion.scalar = args.oim_scalar + (epoch // 5)
        trainer.train(epoch, train_loader, optimizer)
        top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID Training ResNet Model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=['cuhk03', 'market1501', 'viper', 'duke'])
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--num-instances', type=int, default=0,
                        help="If greater than zero, each minibatch will"
                             "consist of (batch_size // num_instances)"
                             "identities, and each identity will have"
                             "num_instances instances. Used in conjunction with"
                             "--loss triplet")
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # loss
    parser.add_argument('--loss', type=str, default='xentropy',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=10)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    parser.add_argument('--triplet-margin', type=float, default=0.5)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
