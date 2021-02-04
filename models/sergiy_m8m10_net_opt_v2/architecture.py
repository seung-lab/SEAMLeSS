import torch
import torch.nn as nn
import copy
import os
import six

import artificery
import torchfields

from utilities.helpers import grid_sample, downsample, downsample_field, load_model_from_dict

from .residuals import res_warp_img, combine_residuals
from .optimizer import optimize_metric

class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, height=3, mips=(8, 10), *args, **kwargs):
        super().__init__()
        self.height = height
        self.mips = mips
        self.range_adjust = False
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        a = artificery.Artificery(device=self.device)
        path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_folder_path = os.path.join(path, 'checkpoint')
        spec_path = os.path.join(checkpoint_folder_path, "model_spec.json")
        my_p = a.parse(spec_path)
        name = 'model'
        checkpoint_path = os.path.join(checkpoint_folder_path, "{}.state.pth.tar".format(name))
        if os.path.isfile(checkpoint_path):
            my_p.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            raise Exception("Weights are missing")
        my_p.name = name
        self.align = my_p

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, src, tgt, src_field=None, tgt_field=None, **kwargs):
        warped_src = src
        with torch.no_grad():
            if src_field is not None and (src_field != 0).sum() > 0:
                src_field *= 0.5 * src.shape[-2]
                warped_src = res_warp_img(src, src_field, is_pix_res=True)
            warped_tgt = tgt
            if tgt_field is not None and (tgt_field != 0).sum() > 0:
                tgt_field *= 0.5 * tgt.shape[-2]
                warped_tgt = res_warp_img(tgt, tgt_field, is_pix_res=True)

            model_run_params = {'level_in': 8}
            stack = torch.cat((warped_src, warped_tgt), 1)

            if 'src_mask' in kwargs:
                src_folds = (kwargs['src_mask'] > 0).float()
            else:
                src_folds = torch.zeros_like(src)

            pred_res = self.align(x=stack, **model_run_params)
            if pred_res.shape[1] == 2:
                pred_res = pred_res.permute(0, 2, 3, 1)

        for k in list(self.align.state.keys()):
            del self.align.state[k]

        pred_res = combine_residuals(pred_res, src_field, is_pix_res=True)
        final_res = optimize_metric(None, src, tgt, pred_res,
                    src_small_defects=torch.zeros_like(src),
                    src_large_defects=src_folds.float(),
                    src_defects=torch.zeros_like(src),
                    tgt_defects=torch.zeros_like(src),
                    max_iter=100)

        final_res = final_res * 2 / src.shape[-2]
        return final_res
