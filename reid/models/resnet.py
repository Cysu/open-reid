import math

import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, \
    resnet152


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_classes=0, num_features=0, norm=False, dropout=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_classes = num_classes
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0

            # Remove the last fc layer (replace it by an identity)
            out_planes = self.base.fc.in_features
            self.base.fc = Identity()

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        if self.cut_at_pooling:
            for name, module in self.base._modules.items():
                if name == 'avgpool':
                    break
                x = module(x)
            return x

        x = self.base(x)
        x = x.view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = x / x.norm(2, 1).expand_as(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
