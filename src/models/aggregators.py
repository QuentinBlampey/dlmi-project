import torch
import torch.nn as nn

class MeanAggregator(nn.Module):
    def forward(self, x):
        return torch.mean(x, axis=0)

class DotAttentionAggregator(nn.Module):
    def __init__(self, size):
        super(DotAttentionAggregator, self).__init__()
        self.fc = nn.Linear(size, 1)

    def forward(self, x):
        """
        x: a (n_samples * size) tensor
        returns: a (size) tensor
        """
        attention = self.fc(x)
        return x.T.mm(attention).view(-1)
