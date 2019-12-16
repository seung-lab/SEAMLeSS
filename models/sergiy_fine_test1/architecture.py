import torch
import torch.nn as nn
import copy
import os

import artificery
from utilities.helpers import gridsample_residual
import torchfields

from .preprocessors import RangeAdjuster, TissueNormalizer
from .residuals import res_warp_img, combine_residuals
from .optimizer import optimizer
from .pre_post_optimizer import pre_post_multiscale_optimizer
from .pre_post_optimizer_multiscale import optimize_metric

class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.range_adjuster = RangeAdjuster(divide=1)
        self.tissue_normalizer = TissueNormalizer()

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

    def forward(self, src, tgt, in_field=None, plastic_mask=None, mip_in=4,
                encodings=False, **kwargs):
        model_run_params = {'level_in': mip_in}
        adj_res = None
        src_folds = (src < 0.35).float()
        tgt_folds = (tgt < 0.35).float()
        src = self.tissue_normalizer(self.range_adjuster(src), tgt_folds)
        tgt = self.tissue_normalizer(self.range_adjuster(tgt), tgt_folds)
        stack = torch.cat((src, tgt), 1)

        pred_res = self.align(x=stack, **model_run_params)
        if pred_res.shape[1] == 2:
            pred_res = pred_res.permute(0, 2, 3, 1)

        final_res = pred_res
        if src.var() > 1e-4:
            final_res = optimize_metric(self.align, src, tgt, final_res, src_folds, tgt_folds)
            pass
        else:
            print ("skipping fucking shit")
        #final_res[..., 0] = src_folds
        #final_res[..., 1] = src_folds

        #final_res[..., 0] = 0
        #final_res[..., 1] = 0
        #field = self.align(stack, mip_in=mip_in)
        final_res = final_res * 2 / src.shape[-2]
        #final_res = final_res.transpose(1, 2)
        return final_res
