import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, ConcatDataset
from normalizer import Normalizer
from aug import aug_stacks, aug_input, rotate_and_scale, crack, displace_slice
import numpy as np
import random

def compile_dataset(h5_paths, transform=None):
    datasets = []
    for h5_path in h5_paths:
        datasets.extend(h5_to_dataset_list(h5_path, transform=transform))
    return ConcatDataset(datasets)

def h5_to_dataset_list(h5_path, transform=None):
    """Create a list of StackDatasets from H5 file created by gen_stack.py
    """
    stacks = []
    h5f = h5py.File(h5_path, 'r')
    for k in h5f.keys():
        d = h5f[k]
        for i in range(d.shape[0]):
            stacks.append(StackDataset(d[i:i+1], transform=transform))
    return stacks


class RandomRotateAndScale(object):
    """Randomly rotate & scale src and tgt images
    """
    def __call__(self, X):
        src, tgt = X['src'], X['tgt']
        if random.randint(0, 1) == 0:
            src, grid = rotate_and_scale(src, None)
            tgt = rotate_and_scale(tgt, grid=grid)[0].squeeze()
            # if src_mask is not None:
            #     src_mask = torch.ceil(
            #         rotate_and_scale(
            #             src_mask.unsqueeze(0).unsqueeze(0), grid=grid
            #         )[0].squeeze())
            #     tgt_mask = torch.ceil(
            #         rotate_and_scale(
            #             tgt_mask.unsqueeze(0).unsqueeze(0), grid=grid
            #         )[0].squeeze())

        src = src.squeeze()
        tgt = tgt.squeeze()
        return {'src': src, 'tgt': tgt}

class Normalize(object):
    """Normalize range of image
    """
    def __init__(self, mip=2):
        self.normalize = Normalizer(mip)

    def __call__(self, X):
        src, tgt = X['src'], X['tgt']
        src = self.normalize.apply(src)
        tgt = self.normalize.apply(tgt)
        return {'src': src, 'tgt': tgt}

class ToFloatTensor(object):
    """Convert ndarray to FloatTensor
    """
    def __call__(self, X):
        src, tgt = X['src'], X['tgt']
        src = torch.FloatTensor(src) / 255.
        tgt = torch.FloatTensor(tgt) / 255.
        return {'src': src, 'tgt': tgt}


class StackDataset(Dataset):
    """Deliver consecutive image pairs from 3D image stack 

    Args:
        stack (4D ndarray): 1xZxHxW image array
    """

    def __init__(self, stack, transform=None):
        self.stack = stack
        self.transform = transform

    def __len__(self):
        # N-1 consecutive image pairs
        return self.stack.shape[1]-1

    def __getitem__(self, k):
        src = self.stack[0, k, :, :]
        tgt = self.stack[0, k+1, :, :]
        X = {'src': src, 'tgt': tgt}
        if self.transform:
            X = self.transform(X)
        return X
