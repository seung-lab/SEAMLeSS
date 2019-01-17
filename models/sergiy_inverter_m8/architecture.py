import torch
import torch.nn as nn
import copy
from utilities.helpers import gridsample_residual, upsample, downsample, load_model_from_dict
from alignermodule import Aligner
from rollback_pyramid import RollbackPyramid


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, height=1, mips=(8), *args, **kwargs):
        super().__init__()
        self.height = height
        self.mips = mips
        self.encode = None
        self.invert = None
        self.aligndict = {}
        self.upsampler = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsampler = torch.nn.AvgPool2d(2, count_include_pad=False)

    def __getitem__(self, index):
        return self.submodule(index)

    def upsample_residual(self, res):
        result = self.upsampler(res.permute(
                                0, 3, 1, 2)).permute(0, 2, 3, 1)
        result *= 2
        return result

    def downsample_residual(self, res):
        result = self.downsampler(res.permute(
                                0, 3, 1, 2)).permute(0, 2, 3, 1)
        result /= 2
        return result

    def forward(self, field, **kwargs):
        field_pixres = field * src.shape[-2] / 2

        field_pixres_downs = field_pixres
        for _ in range(6, 8):
            field_pixres_downs = self.downsample_residual(field_pixres_downs)

        inv_field_pixres_downs = self.invert(field_pixres_downs)

        inv_field_pixres = inv_field_pixres_downs
        for _ in range(6, 8):
            inv_field_pixres = self.downsample_residual(inv_field_pixres)

        inv_field = inv_field_pixres * 2 / src.shape[-2]
        print (loss(field, inv_field))
        return inv_field

    def loss(self, pred_res, inv_res):
        f = combine_residuals(pred_res, inv_res)
	g = combine_residuals(inv_res, pred_res)
	if is_pix_res:
	    f = 2 * f / (f.shape[-2])
	    g = 2 * g / (g.shape[-2])
	loss = 0.5 * torch.mean(f**2) + 0.5 * torch.mean(g**2)
	return loss

    def load(self, path):
        """
        Loads saved weights into the model
        """
        fms = 24
        self.invert = Aligner(fms=[2, fms, fms, fms, fms, 2], k=7).cuda()
        with (path/'inverter_mip8{}.pth.tar'.format(m)).open('rb') as f:
            self.invert.load_state_dict(torch.load(f))
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
