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
        self.forward = self.encode.forward

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
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=True)


def maxpool2x2():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


class UpConv2x2(nn.Module):
    def __init__(self, channels):
        super(UpConv2x2, self).__init__()
        self.conv = nn.Conv2d(channels, channels // 2,
            kernel_size=2, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.pad(x, (0,1,0,1))
        x = self.conv(x)
        return x


def concat(xh, xv):
    return torch.cat([xh, xv], dim=1)


# unet blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: number of channels in input (1st) feature map
            out_channels: number of channels in output feature maps
        """
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: number of channels in input (1st) feature map
            out_channels: number of channels in output feature maps
        """
        super(DownConvBlock, self).__init__()
        self.maxpool = maxpool2x2()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: number of channels in input (1st) feature map
            out_channels: number of channels in output feature maps
        """
        super(UpConvBlock, self).__init__()
        self.upconv = UpConv2x2(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, xh, xv):
        """
        Args:
            xh: torch Variable, activations from same resolution feature maps (gray arrow in diagram)
            xv: torch Variable, activations from lower resolution feature maps (green arrow in diagram)
        """
        xv = self.upconv(xv)
        x = concat(xh, xv)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        fs = [16,32,64,128,256]
        self.conv_in = ConvBlock(1, fs[0])
        self.dconv1 = DownConvBlock(fs[0], fs[1])
        self.dconv2 = DownConvBlock(fs[1], fs[2])
        self.dconv3 = DownConvBlock(fs[2], fs[3])
        self.dconv4 = DownConvBlock(fs[3], fs[4])

        self.uconv1 = UpConvBlock(fs[4], fs[3])
        self.uconv2 = UpConvBlock(fs[3], fs[2])
        self.uconv3 = UpConvBlock(fs[2], fs[1])
        self.uconv4 = UpConvBlock(fs[1], fs[0])
        self.conv_out = conv3x3(fs[0], 2)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.dconv1(x1)
        x3 = self.dconv2(x2)
        x4 = self.dconv3(x3)
        x5 = self.dconv4(x4)
        x6 = self.uconv1(x4, x5)
        x7 = self.uconv2(x3, x6)
        x8 = self.uconv3(x2, x7)
        x9 = self.uconv4(x1, x8)
        x10 = self.conv_out(x9)
        return x10

    def _initialize_weights(self):
        conv_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        for m in conv_modules:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
