import torch
import torch.nn as nn
import copy
from utilities.helpers import normxcorr2, is_blank
import numpy as np


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __getitem__(self, index):
        return None

    def forward(self, src, tgt, **kwargs):
        r = normxcorr2(src, tgt, mode='full')
        if is_blank(r):
          x, y = 0, 0
        elif torch.isnan(r[0,0,0,0]):
          x, y = 0, 0
        else:
          idx = torch.argmax(r)
          x = idx // r.shape[-2]
          y = idx % r.shape[-1]
          x = x - np.ceil(r.shape[-2] / 2)
          y = y - np.ceil(r.shape[-1] / 2)
        return torch.Tensor([[[[x, y]]]], device=src.device)
