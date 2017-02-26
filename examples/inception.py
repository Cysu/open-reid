from __future__ import print_function
import argparse
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from reid.datasets import get_dataset
from reid.loss.oim import OIMLoss
from reid.models import Inception
from reid.train import Trainer, Evaluator
from reid.utils.data import transforms
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_model_, save_model


def get_data(dataset_name, data_dir, batch_size, workers):
    root = osp.join(data_dir, dataset_name)
    dataset = get_dataset(dataset_name, root,
                          split_id=0, num_val=100, download=True)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RandomSizedRectCrop(144, 56),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=False)

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

    # Create data loaders
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.data_dir, args.batch_size, args.workers)

    # Create model
    if args.loss == 'xentropy':
        model = Inception(num_classes=dataset.num_train_ids,
                          num_features=256, dropout=0.5)
    else:
        model = Inception(num_features=256, norm=True, dropout=0.5)
    model = torch.nn.DataParallel(model).cuda()

    # Load from checkpoint
    if args.resume:
        checkpoint = load_model_(args.resume, model)
        args.start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> start epoch {}  best top1 {:.1%}"
              .format(args.start_epoch, best_top1))
    else:
        best_top1 = 0

    cudnn.benchmark = True

    # Evaluator
    evaluator = Evaluator(model, args)
    if args.evaluate:
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    # Criterion and optimizer
    if args.loss == 'xentropy':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = OIMLoss(256, dataset.num_train_ids, scalar=20).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion, args)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 ** (epoch // 50))
        for g in optimizer.param_groups:
            g['lr'] = lr

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_model({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best)

    # Final test
    print('Test:')
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID Training Inception Model")
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=['cuhk03', 'viper'])
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--loss', type=str, default='xentropy',
                        choices=['xentropy', 'oim'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--logs-dir', type=str, default='logs')
    main(parser.parse_args())
