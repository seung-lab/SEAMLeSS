#FIXME test Unet
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

class Unet(nn.Module):
    def __init__(self, eps=0.1, kernel_shape = [[3,3,2,8],
                                                [3,3,8,16],
                                                [3,3,16,32],
                                                [3,3,32,64]]):
        super(Unet, self).__init__()
        self.blocks = []
        levels = len(kernel_shape)
        self.downsamples = nn.ModuleList()
        for i in range(levels-1):
            self.downsamples.append(self.block(kernel_shape[i]))

        self.mid = self.block(kernel_shape[-1])
        self.final = nn.Conv2d(kernel_shape[0][3], kernel_shape[0][2],
                               kernel_size=kernel_shape[0][0], padding=1)
        self.final.weight.data *= eps
        self.final.bias.data *= 0
        self.levels = levels
        self.upsamples = []
        for i in range(1, levels):
            self.upsamples.append(self.block(kernel_shape[-i], down=True))

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    #FIXME add resnet if needed
    def block(self, shape, padding=1, down=False):
        inp, out = shape[2], shape[3]
        if down:
            inp, out = shape[3], shape[2]

        return nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=shape[0], padding=padding),
                nn.ReLU(True),
                nn.Conv2d(out, out, kernel_size=shape[0], padding=padding),
                nn.ReLU(True),
                nn.Conv2d(out, out, kernel_size=shape[0], padding=padding),
                nn.ReLU(True),
                ).cuda()

    def forward(self, x):
        layers = [x]
        for i in range(self.levels-1):
            x = self.downsamples[i](x)
            layers.append(x)
            x = self.pool(x)

        x = self.mid(x)
        layers.append(x)

        for i in range(1, self.levels):
            x = self.upsample(x)
            x_enc = layers[(self.levels-1)-(i-1)]
            x = self.upsamples[i-1](x)
            x = x+x_enc
        x = self.final(x)

        return x
