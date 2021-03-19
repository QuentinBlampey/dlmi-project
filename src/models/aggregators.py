import torch
import torch.nn as nn
import torch.nn.functional as F

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
        attention = F.softmax(attention, dim=0)
        return x.T.mm(attention).view(-1)
