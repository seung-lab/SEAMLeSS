import torch
import torch.nn as nn
import copy
import os

import artificery
from utilities.helpers import gridsample_residual
from .residuals import res_warp_img, combine_residuals

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
        self.range_adjust = True

        a = artificery.Artificery()
        path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_folder_path = os.path.join(path, 'checkpoint')
        spec_path = os.path.join(checkpoint_folder_path, "model_spec.json")
        my_p = a.parse(spec_path)
        name = 'model'
        checkpoint_path = os.path.join(checkpoint_folder_path, "{}.state.pth.tar".format(name))
        if os.path.isfile(checkpoint_path):
            my_p.load_state_dict(torch.load(checkpoint_path))
        else:
            raise Exception("Weights are missing")
        my_p.name = name
        self.align = my_p

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, src, tgt, in_field=None, plastic_mask=None, mip_in=8,
                encodings=False, **kwargs):
        model_run_params = {'level_in': mip_in}
        stack = torch.cat((src, tgt), 1)
        adj_res = None
        if self.range_adjust:
            pred_res_adj = self.align(stack, **model_run_params)
            if pred_res_adj.shape[1] == 2:
                pred_res_adj = pred_res_adj.permute(0, 2, 3, 1)
            x_res = pred_res_adj[tissue_mask][..., 0]
            y_res = pred_res_adj[tissue_mask][..., ]
            x_mean = x_res.mean()
            y_mean = y_res.mean()
            x_max, x_min = x_res.max(), x_res.min()
            y_max, y_min = y_res.max(), y_res.min()
            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            #print ("Adjustment -- X mean: {}, Y mean: {}".format(x_mid, y_mid))
            #print ("Alternative -- X mean: {}, Y mean: {}".format(x_mean, y_mean))
            #print ("Bounds -- X min: {}, X max: {}, X mid: {}".format(x_min, x_max, x_mid))
            #print ("Bounds -- Y min: {}, Y max: {}, Y mid: {}".format(y_min, y_max, y_mid))
            adj_res = torch.ones_like(pred_res_adj)
            adj_res[..., 0] = adj_res[..., 0] * x_mid
            adj_res[..., 1] = adj_res[..., 1] * y_mid
            adj_src = res_warp_img(src, adj_res, is_pix_res=True)
            stack = torch.cat((adj_src, tgt), 1)
        pred_res = self.align(x=stack, **model_run_params)
        if pred_res.shape[1] == 2:
            pred_res = pred_res.permute(0, 2, 3, 1)
        #pred_tgt = res_warp_img(src, pred_res, is_pix_res=True)
        #import pdb; pdb.set_trace()

        if adj_res is not None:
            final_res = combine_residuals(pred_res, adj_res, is_pix_res=True)
        else:
            final_res = pred_res

        #field = self.align(stack, mip_in=mip_in)
        final_res = final_res * 2 / src.shape[-2]
        #final_res = final_res.transpose(1, 2)
        return final_res
