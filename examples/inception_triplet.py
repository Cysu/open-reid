from __future__ import print_function
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from reid.datasets import get_dataset
from reid.mining import mine_hard_triplets
from reid.models import InceptionNet
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import TripletNet
from reid.train_triplet import Trainer, Evaluator
from reid.utils.data import transforms
from reid.utils.data.sampler import RandomTripletSampler, SubsetRandomSampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, \
    copy_state_dict


def get_data(dataset_name, split_id, data_dir, batch_size, workers):
    root = osp.join(data_dir, dataset_name)
    dataset = get_dataset(dataset_name, root,
                          split_id=split_id, num_val=100, download=True)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RandomSizedRectCrop(144, 56),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        sampler=RandomTripletSampler(dataset.train),
        batch_size=batch_size, num_workers=workers, pin_memory=False)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(144, 56),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(144, 56),
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
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.workers)

    # Create models
    base_model = InceptionNet(num_classes=0, num_features=args.features,
                              norm=True, dropout=args.dropout)
    embed_model = EltwiseSubEmbed()
    model = TripletNet(base_model, embed_model)
    model = torch.nn.DataParallel(model).cuda()

    if args.retrain:
        checkpoint = load_checkpoint(args.retrain)
        copy_state_dict(checkpoint['state_dict'], base_model, strip='module.')
        copy_state_dict(checkpoint['state_dict'], embed_model, strip='module.')

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
    evaluator = Evaluator(base_model, embed_model, args)
    if args.evaluate:
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    if args.hard_examples:
        # Use sequential train set loader
        data_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir,
                         transform=val_loader.dataset.transform),
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, pin_memory=False)
        # Mine hard triplet examples, index of [(anchor, pos, neg), ...]
        triplets = mine_hard_triplets(torch.nn.DataParallel(base_model).cuda(),
                                      data_loader, margin=args.margin)
        print("Mined {} hard example triplets".format(len(triplets)))
        # Build a hard examples loader
        train_loader.sampler = SubsetRandomSampler(triplets)

    # Criterion
    criterion = torch.nn.MarginRankingLoss(margin=args.margin).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion, args)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 ** (epoch // 40))
        for g in optimizer.param_groups:
            g['lr'] = lr

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
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
    parser = argparse.ArgumentParser(
        description="Training Inception Siamese Model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=['cuhk03', 'market1501', 'viper'])
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--hard-examples', action='store_true')
    # model
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # loss
    parser.add_argument('--margin', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--retrain', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())