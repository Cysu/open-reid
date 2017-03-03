from torch import nn


class Siamese(nn.Module):
    def __init__(self, base_model, embed_model):
        super(Siamese, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1, x2):
        x1, x2 = self.base_model(x1), self.base_model(x2)
        if self.embed_model is None:
            return x1, x2
        return self.embed_model(x1, x2)
