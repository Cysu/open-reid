from torch import nn


class EltwiseSubEmbed(nn.Module):
    def __init__(self, num_features, nonlinearity='square'):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.bn = nn.BatchNorm1d(num_features)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.classifier = nn.Linear(num_features, 2)
        self.classifier.weight.data.normal_(0, 0.001)
        self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        x = self.bn(x)
        x = self.classifier(x)
        return x
