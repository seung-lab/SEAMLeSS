from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
import os
import math

from .residuals import res_warp_img, res_warp_res

class Aligner(nn.Module):
    def initc(self, m, mult=1):
        m.weight.data *= mult

    def __init__(self, fms=[2, 16, 2], k=7, initc_mult=1.0):
        super(Aligner, self).__init__()
        self.best_val = math.inf

        p = (k - 1) // 2
        self.layers = []

        for i in range(len(fms) - 1):
            self.layers.append(nn.Conv2d(fms[i], fms[i + 1], k, padding=p))
            self.initc(self.layers[-1], initc_mult)

            if i != len(fms) - 2:
                self.layers.append(nn.LeakyReLU(inplace=True))

        self.seq = nn.Sequential(*self.layers)

    def forward_no_perm(self, x):
        return self.seq(x)

    def forward(self, x):
        return self.seq(x).permute(0, 2, 3, 1)

    def run_pair(self, sample):
        sample_var = Variable(sample.cuda(), requires_grad=True)

        src_var = sample_var[:, 0:1, :, :]
        tgt_var = sample_var[:, 1:2, :, :]
        true_res = sample_var[:, 2:4, :, :].permute(0, 2, 3, 1)

        pred_res = self.forward(sample_var[:, 0:2, :, :])
        pred_tgt = res_warp_img(src_var, pred_res, is_pix_res=True)

        return src_var, tgt_var, true_res, pred_res, pred_tgt

    def get_all_params(self):
        params = []
        params.extend(self.parameters())
        return params
