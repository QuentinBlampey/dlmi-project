import torch
import torch.nn as nn


class BaselineNN(nn.Module):
    def __init__(self):
        super(BaselineNN, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 9, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(9, 12, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.linear = nn.Sequential(
            nn.Linear(12 * 26 * 26, 1),
        )

    def forward(self, images):
        x = self.convnet(images)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = torch.mean(x, dim=0)
        # x = F.log_softmax(x, dim=0)
        x = torch.sigmoid(x)
        return x
