import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.helpers import load_model_from_dict


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encode = UNet()
        self.forward = self.encode.discrim

    def __getitem__(self, index):
        return self.submodule(index)

    def __len__(self):
        return self.height

    def load(self, path):
        """
        Loads saved weights into the model
        """
        with path.open('rb') as f:
            weights = torch.load(f)
        load_model_from_dict(self.encode, weights)
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        with path.open('wb') as f:
            torch.save(self.encode.state_dict(), f)

    @property
    def height(self):
        return 1

    def submodule(self, index):
        """
        Returns a submodule as indexed by `index`.

        Submodules with lower indecies are intended to be trained earlier,
        so this also decides the training order.

        `index` must be an int, a slice, or None.
        If `index` is a slice, the submodule contains the relevant levels.
        If `index` is None or greater than the height, the submodule
        returned contains the whole model.
        """
        return self

    def train_level(self, *args, **kwargs):
        return self

    def init_level(self, *args, **kwargs):
        return self


# helper operations
def conv3x3x3(in_channels, out_channels):

    result = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True)

    return result
    

def maxpool2x2():

    result = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)

    return result


def maxpool2x2x2():

    result = nn.MaxPool3d(kernel_size=(2,2,2), stride=2, padding=0)

    return result


class UpConv2x2(nn.Module):
    def __init__(self, channels):

        super(UpConv2x2, self).__init__()
        self.conv = nn.Conv3d(channels, channels//2, kernel_size=(1,1,1), stride=1, padding=0, bias=True)


    def forward(self, x):

        x = F.interpolate(x, scale_factor=(1,2,2), mode='trilinear')
        x = self.conv(x)

        return x


def concat(xh, xv):

    return torch.cat([xh, xv], dim=1) 


class ConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3, self).__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm3d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class DownConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock3, self).__init__()
        self.downsample = maxpool2x2()
        self.convblock = ConvBlock3(in_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.convblock(x)

        return x


class UpConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock3, self).__init__()
        self.upsample = UpConv2x2(in_channels)
        self.convblock = ConvBlock3(in_channels//2 + out_channels, out_channels)

    def forward(self, xh, xv):
        xv = self.upsample(xv)
        x = concat(xh, xv)
        x = self.convblock(x)

        return x


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvOut, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv(x)

        return x


# Architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        fs = [16,32,64,128,256,512]
        self.conv_in = ConvBlock3(2, fs[0])
        self.dconv1 = DownConvBlock3(fs[0], fs[1])
        self.dconv2 = DownConvBlock3(fs[1], fs[2])
        self.dconv3 = DownConvBlock3(fs[2], fs[3])
        self.dconv4 = DownConvBlock3(fs[3], fs[4])
        self.dconv5 = DownConvBlock3(fs[4], fs[5])

        self.uconv1 = UpConvBlock3(fs[5], fs[4])
        self.uconv2 = UpConvBlock3(fs[4], fs[3])
        self.out2 = ConvOut(fs[3], 1)
        self.uconv3 = UpConvBlock3(fs[3], fs[2])
        self.uconv4 = UpConvBlock3(fs[2], fs[1])
        self.uconv5 = UpConvBlock3(fs[1], fs[0])
        self.conv_out = conv3x3x3(fs[0], 1)

        self._initialize_weights()

    def forward(self, x):

        x = self.conv_in(x)
      
        d1 = self.dconv1(x)
        d2 = self.dconv2(d1)       
        d3 = self.dconv3(d2)      
        d4 = self.dconv4(d3)
        d5 = self.dconv5(d4) 

        u1 = self.uconv1(d4, d5)
        u2 = self.uconv2(d3, u1) 
        u3 = self.uconv3(d2, u2) 
        u4 = self.uconv4(d1, u3)
        u5 = self.uconv5(x, u4)     

        reconstruct = self.conv_out(u5)

        return reconstruct

    def discrim(self, x):

        x = self.conv_in(x)
      
        d1 = self.dconv1(x)
        d2 = self.dconv2(d1)       
        d3 = self.dconv3(d2)      
        d4 = self.dconv4(d3)
        d5 = self.dconv5(d4)

        u1 = self.uconv1(d4, d5)
        u2 = self.uconv2(d3, u1)
        out2 = self.out2(u2)

        return out2

    def _initialize_weights(self):
        conv_modules = [m for m in self.modules() if isinstance(m, nn.Conv3d)]
        for m in conv_modules:
            n = m.weight.shape[1]*m.weight.shape[2]*m.weight.shape[3]*m.weight.shape[4]
            m.weight.data.normal_(0, np.sqrt(2. / n))
            m.bias.data.zero_()

