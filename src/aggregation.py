import torch
import torch.nn as nn

class MeanAggregation():
    def __call__(self, x):
        return torch.mean(x, axis=0)

class DotAttentionAggregation(nn.Module):
    def __init__(self, size):
        super(DotAttentionAggregation, self).__init__()
        self.fc = nn.Linear(size, 1)

    def __call__(self, x):
        """
        x: a (n_samples * size) tensor
        returns: a (size) tensor
        """
        attention = self.fc(x)
        return x.T.mm(attention).view(-1)
