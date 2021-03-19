import torch
import torch.nn as nn


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


class LinearHead(nn.Module):
    def __init__(self, size):
        super(LinearHead, self).__init__()
        self.input_size = size + 3
        self.linear = nn.Linear(self.input_size, 1)

    def forward(self, x, medical_data):
        """
        x: a (size) tensor
        medical_data: a (3) tensor
        returns: a scalar tensor
        """
        x = torch.cat((x, medical_data))
        x = self.linear(x)
        return x


class GatedHead(nn.Module):
    def __init__(self, size):
        super(GatedHead, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(size + 3, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.medical_model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cnn_model = nn.Sequential(
            nn.Linear(size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, medical_data):
        pi_cnn = self.gate(torch.cat((x, medical_data)))
        y_med = self.medical_model(medical_data)
        y_cnn = self.cnn_model(x)
        return y_med, y_cnn, pi_cnn * y_cnn + (1 - pi_cnn) * y_med
