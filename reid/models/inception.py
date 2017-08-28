from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


__all__ = ['InceptionNet', 'inception']


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, pool_method, stride):
        super(Block, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                _make_conv(in_planes, out_planes, kernel_size=1, padding=0),
                _make_conv(out_planes, out_planes, stride=stride)
            ),
            nn.Sequential(
                _make_conv(in_planes, out_planes, kernel_size=1, padding=0),
                _make_conv(out_planes, out_planes),
                _make_conv(out_planes, out_planes, stride=stride))
        ])

        if pool_method == 'Avg':
            assert stride == 1
            self.branches.append(
                _make_conv(in_planes, out_planes, kernel_size=1, padding=0))
            self.branches.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                _make_conv(in_planes, out_planes, kernel_size=1, padding=0)))
        else:
            self.branches.append(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1)


class InceptionNet(nn.Module):
    def __init__(self, cut_at_pooling=False, num_features=256, norm=False,
                 dropout=0, num_classes=0):
        super(InceptionNet, self).__init__()
        self.cut_at_pooling = cut_at_pooling

        self.conv1 = _make_conv(3, 32)
        self.conv2 = _make_conv(32, 32)
        self.conv3 = _make_conv(32, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.in_planes = 32
        self.inception4a = self._make_inception(64, 'Avg', 1)
        self.inception4b = self._make_inception(64, 'Max', 2)
        self.inception5a = self._make_inception(128, 'Avg', 1)
        self.inception5b = self._make_inception(128, 'Max', 2)
        self.inception6a = self._make_inception(256, 'Avg', 1)
        self.inception6b = self._make_inception(256, 'Max', 2)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            self.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.has_embedding:
                self.feat = nn.Linear(self.in_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            else:
                # Change the num_features to CNN output channels
                self.num_features = self.in_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)

        self.reset_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception6a(x)
        x = self.inception6b(x)

        if self.cut_at_pooling:
            return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def _make_inception(self, out_planes, pool_method, stride):
        block = Block(self.in_planes, out_planes, pool_method, stride)
        self.in_planes = (out_planes * 4 if pool_method == 'Avg' else
                          out_planes * 2 + self.in_planes)
        return block

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def inception(**kwargs):
    return InceptionNet(**kwargs)
