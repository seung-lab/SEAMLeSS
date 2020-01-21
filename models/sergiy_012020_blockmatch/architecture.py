import torch
import torch.nn as nn

from .blockmatch import block_match

class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tile_step = 96
        self.tile_size = 256
        self.max_disp = 128

    def __getitem__(self, index):
        return None

    def forward(self, src, tgt, in_field=None, plastic_mask=None, mip_in=6,
                encodings=False, **kwargs):
        with torch.no_grad():
            pred_res = block_match(src, tgt, tile_size=self.tile_size,
                                   tile_step=self.tile_step, max_disp=self.max_disp)
        if pred_res.shape[1] == 2:
            pred_res = pred_res.permute(0, 2, 3, 1)
        final_res = pred_res * 2 / src.shape[-2]
        return final_res
