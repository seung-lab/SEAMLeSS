import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

class StackDataset(Dataset):
    def __init__(self, source_h5, mask_h5, basil=False, mask_smooth_factor=None):
        self.dataset = h5py.File(source_h5, 'r')['main']
        self.masks = h5py.File(mask_h5, 'r')['main']
        self.basil = basil
        self.mask_smooth_factor = mask_smooth_factor

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        t = torch.FloatTensor(self.dataset[idx])
        if self.basil:
            zeromask = (t == 0)
            t[t < 150] = 150
            t[t > 200] = 200
            t *= 255.0 / 50
            t = t - torch.min(t) + 1
            t[zeromask] = 0
        t = t / 255.0
        m = torch.FloatTensor(self.masks[idx])
        m_zero_mask = m < 20
        m[m_zero_mask] = 0.001
        m[~m_zero_mask] = 1.0

        if self.mask_smooth_factor is not None:
            m = nn.AvgPool2d(self.mask_smooth_factor,stride=1,padding=(self.mask_smooth_factor-1)//2)(Variable(m)).data

        return t, m
