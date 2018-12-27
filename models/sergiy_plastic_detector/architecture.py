import torch
import torch.nn as nn
import copy
from utilities.helpers import gridsample_residual, upsample, downsample, load_model_from_dict
from masker import Masker


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.height = height
        self.mips = mips
        self.encode = None
        # this is hardcoded here, in a hurry sry
        fms = 24
        self.masker = Masker(fms=[2, fms, fms, fms, fms, 2], k=7).cuda()

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, src, tgt, mip_in=10,
                encodings=False, **kwargs):
        stack = torch.cat((src, tgt), 1)
        mask = self.masker(stack)
        return mask

    def load(self, path):
        """
        Loads saved weights into the model
        """
        with (path/'plastic_detector_mip10.pth.tar').open('rb') as f:
            self.masker.load_state_dict(torch.load(f))
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
