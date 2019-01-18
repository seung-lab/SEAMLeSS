import torch
import torch.nn as nn
import copy
from utilities.helpers import gridsample_residual, upsample, downsample, load_model_from_dict
from scipy.ndimage.filters import gaussian_filter
import math

class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, feature_maps=None, encodings=False, *args, **kwargs):
        super().__init__()
        self.blurrer = self.get_gaussian_kernel(kernel_size=21, sigma=100, channels=2)

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0

    def get_gaussian_kernel(self, kernel_size=3, sigma=2, channels=3):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False,
                                    padding=kernel_size//2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def forward(self, field, **kwargs):
        field_pix = field * field.shape[-2] / 2
        blurred_pix = field_pix  #(field_pix*2).type(torch.cuda.IntTensor).type(torch.cuda.FloatTensor) / 2
        #blurred_pix = self.blurrer(field_pix.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        blurred = blurred_pix * 2 / blurred_pix.shape[-2]
        print (torch.sum(torch.abs(field)))
        print (torch.sum(torch.abs(blurred)))
        #import pdb; pdb.set_trace()
        return field #blurred

    def load(self, path):
        """
        Loads saved weights into the model
        """
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        return

    @property
    def height(self):
        return 0
