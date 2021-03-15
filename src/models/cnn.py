import torch.nn as nn
import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

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
            self.net.classifier[6] = nn.Linear(4096, size)
            # freeze top layers
            for layer in self.net.features:
                for p in layer.parameters():
                    p.requires_grad = False

        elif cnn == 'vgg16':
            self.net = models.vgg16(pretrained=True)
            self.net.classifier.add_module("relu", nn.ReLU(inplace=True))
            self.net.classifier.add_module("last_layer", nn.Linear(1000, size))

            #for name, param in self.net.named_parameters():
            #    if not name.split('.')[0] == 'classifier':
            #        param.requires_grad = False

        elif cnn == 'resnet18':
            self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            # freeze top layers
            for layer in list(self.net.children())[:-10]:
                for param in layer.parameters():
                    param.require_grad = False

            self.net.fc = nn.Linear(512, size)
        
        elif cnn == 'resnet50':
            self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            # freeze top layers
            for layer in list(self.net.children())[:-3]:
                for param in layer.parameters():
                    param.require_grad = False
            self.net.fc = nn.Linear(2048, size)

        elif cnn == 'resnet101':
            self.net = models.resnet101(pretrained=True)
            for layer in list(self.net.children())[:-3]:
                for param in layer.parameters():
                    param.require_grad = False
            self.net.fc = nn.Linear(2048, size)

        elif cnn == 'efficientnet':
            self.net = EfficientNet.from_pretrained('efficientnet-b1', num_classes=size)

        else:
            raise NameError('Invalid cnn name')

    def forward(self, images):
        """
        images: a (n_images * 3 * image_size * image_size) tensor
        returns: a (n_images * size) tensor
        """
        return self.net(images)