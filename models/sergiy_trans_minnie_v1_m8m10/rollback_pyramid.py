import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random
from six import iteritems
import os
import math

from .residuals import res_warp_img, res_warp_res, combine_residuals
from .preprocessors import TissueNormalizer, RangeAdjuster

class RollbackPyramid(nn.Module):
    def __init__(self, max_mip=14, name='pyramid', rollback=True):
        super().__init__()

        self.best_val = math.inf
        self.rollback = rollback
        self.mip_processors = {}
        self.mip_downsamplers = {}
        self.max_mip = max_mip
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)

        self.normalizer = TissueNormalizer()
        self.range_adjuster = RangeAdjuster(divide=1.0)

        for i in range(self.max_mip):
            self.mip_downsamplers[i] = self.default_downsampler()

    def default_downsampler(self):
        return nn.AvgPool2d(2, count_include_pad=False)

    def set_mip_processor(self, module, mip):
        self.mip_processors[mip] = module

    def unset_mip_processor(self, mip):
        del self.mip_processors[mip]

    def set_mip_downsampler(self, module, mip):
        self.mip_downsamplers[mip] = module

    def unset_mip_downsampler(self, mip):
        self.mip_downsamplers[mip] = self.default_downsampler()

    def compute_downsamples(self, img, curr_mip, max_mip):
        downsampled_src_tgt = {}
        downsampled_src_tgt[curr_mip] = img
        for mip in range(curr_mip + 1, max_mip + 1):
            downsampled_src_tgt[mip] = self.mip_downsamplers[mip](
                                                downsampled_src_tgt[mip - 1])
        return downsampled_src_tgt

    def get_all_processor_params(self):
        params = []
        for mip, module in iteritems(self.mip_processors):
            params.extend(module.parameters())
        return params

    def get_processor_params(self, mip):
        params = []
        params.extend(self.mip_processors[mip].parameters())
        return params


    def get_all_downsampler_params(self):
        params = []
        for mip, module in iteritems(self.mip_downsamplers):
            params.extend(module.parameters())
        return params

    def get_downsampler_params(self, mip):
        return [self.mip_downsamplers[mip].parameters()]

    def get_all_params(self):
        params = []
        params.extend(self.get_all_processor_params())
        params.extend(self.get_all_downsampler_params())
        return params

    def upsample_residuals(self, residuals):
        result = self.upsampler(residuals.permute(
                                         0, 3, 1, 2)).permute(0, 2, 3, 1)
        result *= 2

        return result

    def preprocess(self, raw_src_tgt, plastic_mask):
        adjusted_src_tgt = self.range_adjuster(raw_src_tgt)
        final = self.normalizer(adjusted_src_tgt, plastic_mask)
        return final

    def forward(self, raw_src_tgt, plastic_mask, mip_in):
        raw_src_tgt_var = raw_src_tgt.cuda()
        plastic_mask_var = None
        if plastic_mask:
            plastic_mask_var = plastic_mask.cuda()
        src_tgt_var = self.preprocess(raw_src_tgt_var, plastic_mask_var)

        # Find which mips are to be applied
        mips_to_process = iteritems(self.mip_processors)
        filtered_mips   = [m for m in mips_to_process if m[0] >= mip_in]
        ordered_mips    = list(reversed(sorted(filtered_mips)))

        high_mip = ordered_mips[0][0]
        low_mip  = ordered_mips[-1][0]

        # Set up auxilary structures
        residuals  = {}
        downsampled_src_tgt = self.compute_downsamples(src_tgt_var,
                                                       mip_in,
                                                       high_mip)
        # logger.debug("Setting downsample MIP {}".format(high_mip))
        aggregate_res = None
        aggregate_res_mip = None
        # Goal of this loop is to compute aggregate_res
        for mip, module in ordered_mips:
            # generate a warped image at $mip
            if aggregate_res is not None:
                while aggregate_res_mip > mip:
                    aggregate_res = self.upsample_residuals(aggregate_res)
                    aggregate_res_mip -= 1
                if self.rollback:
                    tmp_aggregate_res = aggregate_res
                    for i in range(mip_in, aggregate_res_mip):
                        tmp_aggregate_res = self.upsample_residuals(tmp_aggregate_res)

                    src = res_warp_img(
                                       downsampled_src_tgt[mip_in][:, 0:1],
                                       tmp_aggregate_res)
                    for m in range(mip_in, mip):
                        src = self.mip_downsamplers[m](src)

                else:
                    src = res_warp_img(
                                       downsampled_src_tgt[mip][:, 0:1],
                                       aggregate_res)
            else:
                src = downsampled_src_tgt[high_mip][:, 0:1]

            # Compute residual at level $mip
            tgt = downsampled_src_tgt[mip][:, 1:2]
            sample = torch.cat((src, tgt), 1)
            residuals[mip] = self.mip_processors[mip](sample)

            # initialize aggregate flow if None
            if aggregate_res is None:
                aggregate_res = torch.zeros(residuals[mip].shape, device='cuda')
                aggregate_res_mip = mip

            # Add the residual at level $mip to $aggregate_flow
            aggregate_res = combine_residuals(residuals[mip],
                                              aggregate_res)

        while aggregate_res_mip > mip_in:
            aggregate_res = self.upsample_residuals(aggregate_res)
            aggregate_res_mip -= 1
        return aggregate_res

    def run_pair(self, src_tgt, plastic_mask, mip_in):
        src_tgt_var = Variable(src_tgt.cuda(), requires_grad=True)
        plastic_mask_var= Variable(plastic_mask.cuda(), requires_grad=True)

        src_var = src_tgt_var[:, 0:1, :, :]
        tgt_var = src_tgt_var[:, 1:2, :, :]

        pred_res = self.forward(src_tgt_var, plastic_mask_var, mip_in)
        pred_tgt = res_warp_img(src_var, pred_res, is_pix_res=True)

        return src_var, tgt_var, pred_res, pred_tgt
