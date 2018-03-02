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

class G(nn.Module):
    def __init__(self, skip=False):
        super(G, self).__init__()

        # Spatial transformer localization-network
        kernel_size = 7
        pad = 0
        self.flow = nn.Sequential(
            nn.ReflectionPad2d(15),
            nn.Conv2d(2, 32, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            nn.Conv2d(16,2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
        ).cuda()

        self.skip = skip

    # Flow transformer network forward function
    def forward(self, x, R):
        y = F.grid_sample(x[:,0:1,:,:], R.permute(0,2,3,1), padding_mode='border')
        if self.skip:
            r = torch.zeros_like(R)
            return y, R, r

        r = self.flow(torch.cat([x[:,1:2,:,:], y], dim=1))
        R = r + R
        y = F.grid_sample(x[:,0:1,:,:], R.permute(0,2,3,1),  padding_mode='border')
        return y, R, r

class Pyramid(nn.Module):
    def __init__(self, levels=1, skip_levels=0, shape=[8,2,64,64]):
        super(Pyramid, self).__init__()
        self.G_level = nn.ModuleList()

        for i in range(levels-skip_levels):
            self.G_level.append(G())

        for i in range(skip_levels):
            self.G_level.append(G(skip=True))

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
