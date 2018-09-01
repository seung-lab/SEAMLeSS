import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, ConcatDataset
# from normalizer import Normalizer
import numpy as np

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
    for k in h5f.keys():
        d = h5f[k]
        for i in range(d.shape[0]):
            stacks.append(StackDataset(d[i:i+1]))
    return stacks

class StackDataset(Dataset):
    """Deliver consecutive image pairs from 3D image stack 

    Args:
        stack (4D ndarray): 1xZxHxW image array
    """

    def __init__(self, stack):
        self.stack = stack
        # self.normalizer = Normalizer(2)

    def __len__(self):
        # N-1 consecutive image pairs
        return self.stack.shape[1]-1

    def toFloatTensor(self, image):
        t = torch.FloatTensor(image) / 255.
        t.requires_grad_(False)
        return t 

    def __getitem__(self, k):
        # print('Get {0} / {1}'.format(k, self.stack.shape))
        src = self.toFloatTensor(self.stack[0, k, :, :])
        tgt = self.toFloatTensor(self.stack[0, k+1, :, :])
        # print('sizes, src {0}; tgt {1}'.format(src.size(), tgt.size()))
        X = {'src': src, 'tgt': tgt}
        return X
