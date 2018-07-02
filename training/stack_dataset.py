import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank
from helpers import save_chunk, gif

class StackDataset(Dataset):
    def __init__(self, source_h5, crack_mask_h5, fold_mask_h5, basil=False, threshold_masks=True, combine_masks=True, lm=False):
        self.dataset = h5py.File(source_h5, 'r')['main']
        self.crack_masks = h5py.File(crack_mask_h5, 'r')['main'] if crack_mask_h5 else None
        self.fold_masks = h5py.File(fold_mask_h5, 'r')['main'] if fold_mask_h5 else None
        self.basil = basil
        self.threshold_masks = threshold_masks
        self.combine_masks = combine_masks
        self.dim = self.dataset.shape[-1]
        self.lm = lm

    def __len__(self):
        return self.dataset.shape[0]

    def contrast(self, t, l=145, h=210):
        zeromask = (t == 0)
        t[t < l] = l
        t[t > h] = h
        t *= 255.0 / (h-l+1)
        t = t - np.min(t) + 1
        t[zeromask] = 0
        return t

    def mean_filter(self, chunk, radius=5):
        chunk = chunk.astype(np.uint8)
        return rank.mean(chunk, selem=disk(radius))

    def get_crack_mask(self, idx):
        cm = None
        if self.crack_masks:
            cm = self.crack_masks[idx]
            cm_binary = np.zeros(cm.shape)
            cm_binary[cm < 100] = 1
            cm = cm_binary.astype(np.bool)
        return cm

    def get_fold_mask(self, idx):
        fm = None
        if self.fold_masks:
            fm = self.fold_masks[idx]
            fm_binary = np.zeros(fm.shape)
            fm_binary[fm < 100] = 1
            fm = fm_binary.astype(np.bool)
        return fm

    def thresholded_crack_mask(self, cm):
        cm_px = np.zeros(cm.shape)
        for i in range(cm_px.shape[0]):
            cmi = cm[i].astype(np.uint8) * 255
            cmis = self.mean_filter(cmi, radius=18 if not self.lm else 80) / 255.0
            cmis[cmis < 0.85] = 0
            cmis[cmis >= 0.85] = 1
            cm_px[i] = cmis
        return cm + cm_px

    def thresholded_fold_mask(self, fm):
        fm_px = np.zeros(fm.shape)
        for i in range(fm_px.shape[0]):
            fmi = fm[i].astype(np.uint8) * 255
            fmis = self.mean_filter(fmi, radius=1 if not self.lm else 23) / 255.0
            fmis[fmis < 0.85] = 0
            fmis[fmis >= 0.85] = 1
            fm_px[i] = fmis
            if self.lm: # dilate intermediate mask for low-mip folds
                fmi = self.mean_filter(fmi, radius=150) / 255.0
                fmi[fmi > 0.05] = 1
                fmi[fmi <= 0.05] = 0
                fm[i] = fmi
        return fm + fm_px

    def np_upsample(self, a):
        factor = self.dim // a.shape[-1]
        if factor > 1:
            return a.repeat(factor, axis=-1).repeat(factor, axis=-2)
        else:
            return a

    def __getitem__(self, idx):
        t = self.dataset[idx]
 
        cm = self.get_crack_mask(idx)
        fm = self.get_fold_mask(idx)

        cm = self.np_upsample(cm) if cm is not None else None
        fm = self.np_upsample(fm) if fm is not None else None

        if self.threshold_masks:
            cm = self.thresholded_crack_mask(cm) if cm is not None else None
            fm = self.thresholded_fold_mask(fm) if fm is not None else None

        if cm is None:
            cm = np.zeros(t.shape)
        if fm is None:
            fm = np.zeros(t.shape)

        if self.basil:
            t = self.contrast(t) / 255.0
            
        cm = torch.FloatTensor(cm)
        fm = torch.FloatTensor(fm)
        t = torch.FloatTensor(t)

        # Returns masks of the same size as the image tensor, where a 0 in the mask
        #  indicates unaffected locations, a 1 in the mask indicates the neighborhood impacted
        #  by the crack or fold, and a 2 indicates the location of the crack or fold itself

        if self.combine_masks:
            return {'X': t, 'm': torch.max(cm,fm) }
        else:
            return { 'X': t, 'cm': cm, 'fm': fm }
