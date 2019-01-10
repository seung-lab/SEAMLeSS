import torch
import torch.nn as nn
import copy
from utilities.helpers import gridsample_residual, upsample, downsample, load_model_from_dict
from .alignermodule import Aligner
from .rollback_pyramid import RollbackPyramid


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, height=3, mips=(8, 9, 10), *args, **kwargs):
        super().__init__()
        self.height = height
        self.mips = mips
        self.encode = None
        self.align = RollbackPyramid()
        self.aligndict = {}

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, src, tgt, in_field=None, plastic_mask=None, mip_in=8,
                encodings=False, **kwargs):
        stack = torch.cat((src, tgt), 1)
        if encodings:
            src, tgt = self.encode(src, tgt)
        field = self.align(stack, plastic_mask=None, mip_in=mip_in)
        field = field * 2 / src.shape[-2]
        return field

    def load(self, path):
        """
        Loads saved weights into the model
        """
        for m in self.mips:
            self.aligndict[m] = Aligner(fms=[2, 16, 16, 16, 16, 2], k=7).cuda()
            with (path/'aligner_mip{}.pth.tar'.format(m)).open('rb') as f:
                self.aligndict[m].load_state_dict(torch.load(f))
            self.align.set_mip_processor(self.aligndict[m], m)
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        raise NotImplementedError()
        # with path.open('wb') as f:
        #     torch.save(self.state_dict(), f)

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
        if index is None or (isinstance(index, int)
                             and index >= self.height):
            index = slice(self.height)
        if isinstance(index, int):
            index = slice(index, index+1)
        newmips = range(max(self.aligndict.keys()))[index]
        sub = Model(height=self.height, mips=newmips)
        for m in newmips:
            sub.align.set_mip_processor(self.aligndict[m], m)
        return sub
