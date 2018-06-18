import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

class StackDataset(Dataset):
    def __init__(self, source_h5, crack_mask_h5, fold_mask_h5, basil=False, mask_smooth_factor=None, threshold_masks=False):
        self.dataset = h5py.File(source_h5, 'r')['main']
        self.crack_masks = h5py.File(crack_mask_h5, 'r')['main'] if crack_mask_h5 else None
        self.fold_masks = h5py.File(fold_mask_h5, 'r')['main'] if fold_mask_h5 else None
        self.basil = basil
        self.mask_smooth_factor = mask_smooth_factor
        self.threshold_masks = threshold_masks

    def __len__(self):
        return self.dataset.shape[0]

    def contrast(self, t):
        zeromask = (t == 0)
        t[t < 150] = 150
        t[t > 200] = 200
        t *= 255.0 / 50
        t = t - torch.min(t) + 1
        t[zeromask] = 0
        return t

    def get_crack_mask(self, idx):
        cm = None
        if self.crack_masks:
            cm = torch.FloatTensor(self.crack_masks[idx])
            cm_zero_mask = cm < 20
            cm[cm_zero_mask] = 0.001
            cm[~cm_zero_mask] = 1.0
        return cm

    def get_fold_mask(self, idx):
        fm = None
        if self.fold_masks:
            fm = torch.FloatTensor(self.fold_masks[idx])
            fm_zero_mask = fm < 20
            fm[fm_zero_mask] = 0.001
            fm[~fm_zero_mask] = 1.0
        return fm

    def threshold_crack_mask(self, t, cm):
        return cm
        
    def threshold_fold_mask(self, t, fm):
        return fm

    def __getitem__(self, idx):
        t = torch.FloatTensor(self.dataset[idx])
        if self.basil:
            t = self.contrast(t)
        t = t / 255.0

        cm = self.get_crack_mask(idx)
        fm = self.get_fold_mask(idx)

        if self.threshold_masks:
            cm = self.threshold_crack_mask(t, cm) if cm is not None else None
            fm = self.threshold_fold_mask(t, fm) if fm is not None else None

        if self.mask_smooth_factor is not None:
            cm = nn.AvgPool2d(self.mask_smooth_factor,stride=1,padding=(self.mask_smooth_factor-1)//2)(Variable(cm)).data if cm is not None else None
            fm = nn.AvgPool2d(self.mask_smooth_factor,stride=1,padding=(self.mask_smooth_factor-1)//2)(Variable(fm)).data if fm is not None else None

        if cm is None:
            cm = torch.zeros(t.size())
        if fm is None:
            fm = torch.zeros(t.size())

        return { 'X': t, 'cm': cm, 'fm': fm }
