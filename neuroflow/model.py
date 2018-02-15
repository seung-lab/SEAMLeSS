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
#
# Network that takes input
# - x [batch, 2, width, height] two images
# - R [batch, 2, width, height] coarse estimate of the flow
# returns
# - y [batch, 1, width, height] transformed image
# - R [batch, 2, width, height] finer estimate of the flow
# - r [batch, 2, width, height] residual change to the coarser estimate


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
        self.index = Variable(torch.cuda.LongTensor([0]), requires_grad=False)

    # Flow transformer network forward function
    def forward(self, x, R):
        r = self.flow(x)
        R = r + R
        x = torch.index_select(x, 1, self.index) #FIXME use slicing []
        y = F.grid_sample(x, R.permute(0,2,3,1))
        return y, R, r


### written by Francisco Massa
### Move to Cavelab
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

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

class Pyramid(nn.Module):
    def __init__(self, levels=1, shape=[8,2,64,64]):
        super(Pyramid, self).__init__()
        G_level = []
        for i in range(levels):
            G_level.append(G())
        self.G_level = ListModule(*G_level) #FIXME nn.Module.List

        shape[2], shape[3] = int(shape[2]/2**levels), int(shape[3]/2**levels)
        identity = get_identity(batch_size=shape[0], width=shape[3])

        self.identity = Variable(torch.from_numpy(identity).cuda(), requires_grad=False)

        #self.identity = torch.cuda.FloatTensor(*shape).fill_(0)
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
        return ys, Rs, rs
