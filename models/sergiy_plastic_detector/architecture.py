import torch
import torch.nn as nn
import copy
from .masker import Masker


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        fms = 24
        self.masker = Masker(fms=[1, fms, fms, fms, fms, fms, 1], k=7).cuda()

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, img, mip_in=8,
                encodings=False, **kwargs):
        img = torch.cuda.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255. - 0.5
        img_down   = nn.functional.avg_pool2d(img, (2, 2))
        img_down   = nn.functional.avg_pool2d(img_down, (2, 2))
        img_down_t = img_down#.transpose(2, 3)
        mask_down_t  = self.masker(img_down_t)
        mask_down    = mask_down_t#.transpose(2, 3)
        mask_down    = nn.functional.interpolate(mask_down, scale_factor=2)
        mask         = nn.functional.interpolate(mask_down, scale_factor=2)
        bool_mask    = mask[0, 0].unsqueeze(-1).unsqueeze(-1).cpu().detach().numpy() > 0.9
        return bool_mask

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
