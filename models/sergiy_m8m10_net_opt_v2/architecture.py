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
            if (src_field != 0).sum() > 0:
                warped_src = res_warp_img(src, src_field, is_pix_res=True)
            warped_tgt = tgt
            if (tgt_field != 0).sum() > 0:
                warped_tgt = res_warp_img(tgt, tgt_field, is_pix_res=True)

            model_run_params = {'level_in': 8}
            stack = torch.cat((warped_src, warped_tgt), 1)
            adj_res = None
            if 'src_mask' in kwargs:
                src_folds = (kwargs['src_mask'] > 0).float()
            else:
                src_folds = torch.zeros_like(src)

            if self.range_adjust:
                tissue_mask = (src != 0).squeeze(0).type(torch.cuda.FolatTensor)
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
                x_mid_int = int(x_mid)
                y_mid_int = int(y_mid)

                if x_mid_int == 0 and y_mid_int == 0:
                    adj_res = None
                else:
                    adj_res[..., 0] = adj_res[..., 0] * x_mid_int
                    adj_res[..., 1] = adj_res[..., 1] * y_mid_int
                    #adj_src = res_warp_img(src, adj_res, is_pix_res=True, rollback=2)
                    adj_src = shift_by_int(src, -x_mid_int, -y_mid_int)
                    stack = torch.cat((adj_src, tgt), 1)

            pred_res = self.align(x=stack, **model_run_params)
            if pred_res.shape[1] == 2:
                pred_res = pred_res.permute(0, 2, 3, 1)

            if adj_res is not None:
                pred_res = combine_residuals(pred_res, adj_res, is_pix_res=True, rollback=2)
        for k in list(self.align.state.keys()):
            del self.align.state[k]
        #final_res = pred_res
        final_res = optimize_metric(None, src, tgt, pred_res,
                    src_small_defects=torch.zeros_like(src),
                    src_large_defects=src_folds.float(),
                    src_defects=torch.zeros_like(src),
                    tgt_defects=torch.zeros_like(src),
                    max_iter=100)


        #final_res[..., 0] = src_folds
        #final_res[..., 1] = src_folds

        #final_res[..., 0] = 0
        #final_res[..., 1] = 0
        #field = self.align(stack, mip_in=mip_in)
        final_res = final_res * 2 / src.shape[-2]
        #final_res = final_res.transpose(1, 2)
        return final_res
