import torch.nn as nn
import torch

class FullyConnectedHead(nn.Module):
    def __init__(self, size):
        super(FullyConnectedHead, self).__init__()
        self.input_size = size + 3
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
    
    def forward(self, x, medical_data):
        """
        x: a (size) tensor
        medical_data: a (3) tensor
        returns: a scalar tensor
        """
        x = torch.cat((x, medical_data))
        x = self.fc(x)
        return x