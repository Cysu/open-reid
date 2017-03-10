from __future__ import print_function
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .evaluation import cmc
from .features import extract_features, FeatureDatabase
from .utils.data.sampler import ExhaustivePairSampler
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
        accuracy = AverageMeter()

        self.model.train()

        end = time.time()
        for i, triplet in enumerate(data_loader):
            data_time.update(time.time() - end)

            anchors, positives, negatives, targets = self._parse_data(triplet)

            anchor_neg, anchor_pos = self.model(anchors, positives, negatives)
            loss = self.criterion(anchor_neg, anchor_pos, targets)

            num = anchors.size(0)
            prec = (anchor_neg.data > anchor_pos.data).sum() * 1. / num
            losses.update(loss.data[0], num)
            accuracy.update(prec, num)

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
                      'Accuracy {:.2%} ({:.2%})\t'.format(
                    epoch, i + 1, len(data_loader),
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.avg,
                    losses.val, losses.avg, accuracy.val, accuracy.avg))

    def _parse_data(self, triplet):
        (anchors, _, _, _), (positives, _, _, _), (negatives, _, _, _) = triplet
        anchors, positives, negatives = \
            Variable(anchors), Variable(positives), Variable(negatives)
        targets = Variable(torch.ones(anchors.size(0)).cuda())
        return anchors, positives, negatives, targets


class FeaturePairPreprocessor(object):
    def __init__(self, features, query, gallery):
        super(FeaturePairPreprocessor, self).__init__()
        self.features = features
        self.query = query
        self.gallery = gallery

    def __len__(self):
        return len(self.query) * len(self.gallery)

    def __getitem__(self, index_pair):
        i1, i2 = index_pair
        fname1, _, _ = self.query[i1]
        fname2, _, _ = self.gallery[i2]
        return self.features[fname1], self.features[fname2]


class Evaluator(object):
    def __init__(self, base_model, embed_model, args):
        super(Evaluator, self).__init__()
        self.base_model = torch.nn.DataParallel(base_model).cuda()
        self.embed_model = torch.nn.DataParallel(embed_model).cuda()
        self.args = args

    def evaluate(self, data_loader, query, gallery, cache_file=None):
        # Extract features image by image
        features = extract_features(self.base_model, data_loader,
                                    print_freq=self.args.print_freq,
                                    output_file=cache_file)
        if cache_file is not None:
            features = FeatureDatabase(cache_file, 'r')

        # Build a data loader for exhaustive (query, gallery) pairs
        processor = FeaturePairPreprocessor(features, query, gallery)
        data_loader = DataLoader(
            processor, sampler=ExhaustivePairSampler(query, gallery),
            batch_size=min(len(gallery), 4096), num_workers=self.args.workers,
            pin_memory=False)

        # Do forward of the embedding model
        self.embed_model.eval()

        distmat = []

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        for i, (inputs1, inputs2) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs1 = Variable(inputs1, volatile=True)
            inputs2 = Variable(inputs2, volatile=True)
            outputs = self.embed_model(inputs1, inputs2).data.squeeze()

            assert outputs.numel() == inputs1.size(0)

            distmat.extend(outputs)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                print('Embedding: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'.format(
                    i + 1, len(data_loader),
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.avg))

        if cache_file is not None:
            features.close()

        # Do evaluation
        distmat = np.asarray(distmat).reshape(len(query), len(gallery))

        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]

        # Compute both new and old cmc scores
        cmc_configs = {
            'new': dict(separate_camera_set=False,
                        single_gallery_shot=False,
                        first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True),
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('new', 'cuhk03', 'market1501'))
        for k in [1, 5, 10]:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                  .format(k, cmc_scores['new'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))

        # Use the new cmc top-1 score for validation criterion
        return cmc_scores['new'][0]
