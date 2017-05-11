"""
Cycle GAN:
    Let
        x = some arbitrary value
        y = F(x)
        x = G(y)

    Then
        G(F(x)) = x = G(y)
        F(G(y)) = y = F(x)

Reference: https://arxiv.org/pdf/1703.10593.pdf
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GeneratorCNN(nn.Module):
    """
    Maps domain x to a target domain y using convnets
    """

    def __init__(self, conv_dims, convt_dims):
        super().__init__()

        self.layers = []

        in_channels = conv_dims[0]
        out_channels = convt_dims[-1]
        prev_dim = conv_dims[1]

        self.layers.append(
            nn.Conv2d(in_channels, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[2:]:
            self.layers.append(
                nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        for out_dim in convt_dims[:-1]:
            self.layers.append(nn.ConvTranspose2d(
                prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = out_dim

        self.layers.append(nn.ConvTranspose2d(
            prev_dim, out_channels, 4, 2, 1, bias=False))
        self.layers.append(nn.Tanh())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):

        for layer in self.layer_module:
            x = layer(x)

        return x


class DiscriminatorCNN(nn.Module):
    """
    Determines if the domain is real/fake
    """

    def __init__(self, conv_dims):
        super().__init__()

        self.layers = []

        in_channels = conv_dims[0]
        out_channels = conv_dims[-1]

        prev_dim = conv_dims[1]

        self.layers.append(
            nn.Conv2d(in_channels, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[2:-1]:
            self.layers.append(
                nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        self.layers.append(
            nn.Conv2d(prev_dim, out_channels, 4, 1, 0, bias=False))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):

        for layer in self.layer_module:
            x = layer(x)

        return x
