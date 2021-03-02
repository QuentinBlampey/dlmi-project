import torch.nn as nn
import torch

class BaselineCNN(nn.Module):
    def __init__(self, size=16):
        super(BaselineCNN, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(6 * 111 * 111, size),
        )

    def forward(self, images):
        """
        images: a (n_images * 3 * image_size * image_size) tensor
        returns: a (n_images * size) tensor
        """
        x = self.convnet(images)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class PretrainedCNN(nn.Module):
    def __init__(self, size=16, cnn='vgg11'):
        super(PretrainedCNN, self).__init__()
        if cnn == 'vgg11':
            self.net  = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
            self.net .classifier[6] = nn.Linear(4096, size)
        elif cnn == 'resnet18':
            self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.net.fc = nn.Linear(512, size)
        else:
            raise NameError('Invalid cnn name')

    def forward(self, images):
        """
        images: a (n_images * 3 * image_size * image_size) tensor
        returns: a (n_images * size) tensor
        """
        return self.net(images)