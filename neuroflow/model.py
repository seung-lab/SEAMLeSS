from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

from util import get_identity
import numpy as np

# G - Level convolution
# --------------------------------------
# Network that takes input
# - x [batch, 2, width, height] two images
# - R [batch, 2, width, height] coarse estimate of the flow
# returns
# - y [batch, 1, width, height] transformed image
# - R [batch, 2, width, height] finer estimate of the flow
# - r [batch, 2, width, height] residual change to the coarser estimate

# Pyramid Level
# --------------------------------------
# Initialized with the number of levels
# - levels = 1
# - shape=[batch, 2, width, height] - input shape
# Network that takes input
# - x [batch, 2, width, height] two images
#
# returns the output of each level
# - ys [[batch, 1, width, height]...] transformed image
# - Rs [[batch, 2, width, height]...] finer estimate of the flow
# - rs [[batch, 2, width, height]...] residual change to the coarser estimate

class G_empty(nn.Module):
    def __init__(self, skip=False):
        super(G_empty, self).__init__()
    # Flow transformer network forward function
    def forward(self, x, R):
        return x, R, R

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        # Spatial transformer localization-network
        self.flow = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(16, 2, kernel_size=7, padding=3),
        ).cuda()

    # Flow transformer network forward function
    def forward(self, x, R):
        y = F.grid_sample(x[:,0:1,:,:], R.permute(0,2,3,1))
        r = self.flow(torch.cat([x[:,1:2,:,:], y], dim=1))
        R = r + R
        y = F.grid_sample(x[:,0:1,:,:], R.permute(0,2,3,1))
        return y, R, r

class Pyramid(nn.Module):
    def __init__(self, levels=1, shape=[8,2,64,64]):
        super(Pyramid, self).__init__()
        self.G_level = nn.ModuleList()

        self.G_level.append(G())
        for i in range(levels-1):
            self.G_level.append(G())

        shape[2], shape[3] = int(shape[2]/2**levels), int(shape[3]/2**levels)
        identity = get_identity(batch_size=shape[0], width=shape[3])

        self.identity = Variable(torch.from_numpy(identity).cuda(), requires_grad=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.levels = levels
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        R = self.identity
        xs, ys, Rs, rs = [x], [], [], []

        # Downsample
        for i in range(1, self.levels):
            xs.append(self.downsample(xs[i-1]))
        xs.reverse()

        # Levels
        for i in range(self.levels):
            R = self.upsample(R)
            y, R, r = self.G_level[i](xs[i], R)
            ys.append(y), Rs.append(R), rs.append(r)

        return xs, ys, Rs, rs
