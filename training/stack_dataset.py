import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class StackDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, source_h5):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.dataset = h5py.File(source_h5, 'r')['main']
        
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.dataset[idx] / 255.0)
