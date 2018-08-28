from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

class G(nn.Module):
    def __init__(self, skip=False, eps=0.001, kernel_size=7):
        super(G, self).__init__()

        # Spatial transformer localization-network
        pad = int(kernel_size/2)
        #kernel_size = [kernel_size, kernel_size]
        #pad = (pad, pad)
        self.flow = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            #nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            #nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            #nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 16, kernel_size=kernel_size, padding=pad),
            nn.ReLU(True),
            #nn.BatchNorm2d(16, affine=False),
            nn.Conv2d(16, 2, kernel_size=kernel_size, padding=pad),
            #nn.BatchNorm2d(2),
        ).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal(m.weight)

        #self.flow[-1].weight.data *= eps
        #self.flow[-1].bias.data *= eps

        self.tanh = nn.Tanh()
        self.skip = skip

    # Flow transformer network forward function
    def forward(self, x): #[b,2,256,256]
        if self.skip:
            r = torch.zeros_like(x) #FIXME
            return r
        r = self.flow(x)

        return r #[b, 2, 256, 256]
