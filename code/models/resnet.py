import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, n_features):
        super(ResNet, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        self.linear_1 = nn.Linear(256, n_features)
        self.linear_2 = nn.Linear(n_features, 1)
        # For saving model and loss values
        self.name = "ResNet"

    def forward(self, x):
        h = self.resnet(x)
        h = self.linear_1(h)
        h = self.linear_2(h)

        return h.view(-1)