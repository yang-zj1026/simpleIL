import torch.nn as nn
from utils import available_actions


class Flatten(nn.Module):
    """
    Helper class to flatten the tensor
    between the last conv and first fc layer
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


def build_network():
    """Build the network"""

    model = nn.Sequential(
        nn.Conv2d(1, 32, 8, 4),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Conv2d(32, 64, 4, 2),
        nn.BatchNorm2d(64, 64, 3, 1),
        nn.ReLU(),
        Flatten(),
        nn.BatchNorm1d(64 * 7 * 7),
        nn.Dropout(),
        nn.Linear(64 * 7 * 7, 120),
        nn.ReLU(),
        nn.BatchNorm1d(120),
        nn.Dropout(),
        nn.Linear(120, len(available_actions)),
    )
