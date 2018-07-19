import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import torch
import numpy as np
from torch.utils.data import Dataset

class StackDataset(Dataset):
    def __init__(self, source_h5, mip=-1):
        self.h5f = h5py.File(source_h5, 'r')

        self.datasets = []
        for k in self.h5f.keys():
            if 'pinky' not in k:
                self.datasets.append(self.h5f[k])
            else:
                print('Throwing out pinky...')
        self.lengths = [d.shape[0] for d in self.datasets]
        self.clengths = np.cumsum(self.lengths)
        self.length = sum(self.lengths)
        self.mip = mip

    def map_total_index(self, idx):
        chunks = [idx < l for l in self.clengths]
        assert True in chunks, 'Index {} not contains in Stack Dataset with lengths {}.'.format(idx, self.lengths)
        dataset_idx = chunks.index(True)
        sub_idx = idx if dataset_idx == 0 else idx - self.clengths[dataset_idx - 1]
        return (dataset_idx, sub_idx)
        
    def __len__(self):
        
        return self.length

    def __getitem__(self, idx):
        dataset_idx, sub_idx = self.map_total_index(idx)
        t = torch.FloatTensor(self.datasets[dataset_idx][sub_idx]) / 255.

        output = { 'X': t }
        if self.mip != -1:
            output['mip'] = self.mip

        return output
