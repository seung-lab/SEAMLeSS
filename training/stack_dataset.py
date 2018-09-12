import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, ConcatDataset
from normalizer import Normalizer
from aug import aug_input, rotate_and_scale, random_translation
from helpers import reverse_dim
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

class RandomAugmentation(object):
    """Apply random Gaussian noise, cutouts, & brightness adjustment
    """
    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, X):
        src, tgt = X['src'].clone(), X['tgt'].clone()
        aug_src, aug_src_masks = aug_input(src)
        aug_tgt, aug_tgt_masks = aug_input(tgt)
        X['aug_src'] = aug_src
        X['aug_tgt'] = aug_tgt
        X['aug_src_masks'] = aug_src_masks
        X['aug_tgt_masks'] = aug_src_masks
        return X

class RandomFlip(object):
    """Randomly flip src & tgt images
    """
    def __call__(self, X):
        src, tgt = X['src'], X['tgt']
        if random.randint(0,1) == 0:
            src = reverse_dim(src, 0)
            tgt = reverse_dim(tgt, 0)
        if random.randint(0,1) == 0:
            src = reverse_dim(src, 1)
            tgt = reverse_dim(tgt, 1)
        return {'src': src, 'tgt': tgt}        

class RandomTranslation(object):
    """Randomly translate src & tgt images separately
    """
    def __init__(self, max_displacement=2**6):
        self.max_displacement = max_displacement

    def __call__(self, X):
        src, tgt = X['src'], X['tgt']
        if random.randint(0,1) == 0:
            src = random_translation(src, self.max_displacement)
        if random.randint(0,1) == 0:
            tgt = random_translation(tgt, self.max_displacement)
        return {'src': src, 'tgt': tgt}        

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
    """Normalize range of all tensors
    """
    def __init__(self, mip=2):
        self.normalize = Normalizer(mip)

    def __call__(self, X):
        for k, v in X.items():
            if isinstance(v, torch.Tensor):
                X[k] = torch.FloatTensor(self.normalize.apply(v.numpy()))
        return X 

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
