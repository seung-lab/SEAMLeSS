import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

def compile_dataset(h5_paths):
    datasets = []
    for h5_path in h5_paths:
        datasets.extend(h5_to_dataset_list(h5_path))
    return ConcatDataset(datasets)

def h5_to_dataset_list(h5_path):
    """Create a list of StackDatasets from H5 file created by gen_stack.py
    """
    stacks = []
    h5f = h5py.File(h5_path, 'r')
    for k in h5py.keys:
        d = h5py[k]
        for i in range(d.shape[0])
            stacks.append(StackDataset(d[i:i+1]))
    return stacks

class StackDataset(Dataset):
    """Deliver consecutive image pairs from 3D image stack 

    Args:
        stack (4D ndarray): 1xZxHxW image array
    """

    def __init__(self, stack):
        self.stack = stack

    def __len__(self):
        # N-1 consecutive image pairs
        return self.stack.shape[1]-1

    def toFloatTensor(self, image):
        t = torch.FloatTensor(image) / 255.
        return Variable(t, requires_grad=False).cuda()

    def __getitem__(self, k):
        src = self.toFloatTensor(self.stack[k])
        tgt = self.toFloatTensor(self.stack[k+1])
        X = {'src': src, 'tgt': tgt}
        return X
