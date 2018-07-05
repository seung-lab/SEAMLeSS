import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import torch
import numpy as np

class StackDataset(Dataset):
    def __init__(self, source_h5):
        self.dataset = h5py.File(source_h5, 'r')['main']
        
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        t = self.dataset[idx]
        t = torch.FloatTensor(t)

        return { 'X': t }
