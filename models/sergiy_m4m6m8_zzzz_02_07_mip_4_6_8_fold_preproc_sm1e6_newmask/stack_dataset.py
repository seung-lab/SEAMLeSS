from torch.utils.data import Dataset, ConcatDataset
import copy
import warnings
import torch
import numpy as np
from skimage.transform import resize
#from augment import apply_transform_to_bundle

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

mask_types = ['edges', 'defects', 'plastic']


def compile_dataset(dataset_specs, supervised=True, transform=None,
                    max_size=None):
    datasets = []
    for spec in dataset_specs:
        d = h5_to_dataset_list(spec, supervised=supervised,
                               transform=transform, max_size=max_size)

        datasets.extend(d)
    return ConcatDataset(datasets)


def h5_to_dataset_list(spec, transform=None, supervised=True, max_size=None):
    """Create a list of StackDatasets from H5 file created by gen_stack.py
    """
    stacks = []
    data_h5 = h5py.File(spec['data']['h5'], 'r')

    mask_h5s = {}
    if 'masks' in spec:
        for mask_type in mask_types:
            if mask_type in spec['masks']:
                print ("{} mask detected!".format(mask_type))
                mask_h5s[mask_type] = h5py.File(spec['masks'][mask_type]['h5'], 'r')

    dataset_mip = None
    if 'mip' in spec['data']:
        dataset_mip = spec['data']['mip']

    mask_mips = {}
    for mask_type in mask_types:
        if mask_type in mask_h5s:
            if 'mip' in spec['masks'][mask_type]:
                mask_mips[mask_type] = spec['masks'][mask_type]['mip']

    for k in data_h5.keys():
        d = data_h5[k]

        masks = {}
        for mask_type in mask_types:
            if mask_type in mask_h5s:
                masks[mask_type] = mask_h5s[mask_type][k]

        for i in range(d.shape[0]):
            stacks.append(StackDataset(d[i:i+1], masks=masks, mask_mips=mask_mips,
                          dataset_mip=dataset_mip,
                          transform=transform,
                          supervised=supervised, max_size=max_size))
    return stacks


class StackDataset(Dataset):
    """Deliver consecutive image pairs from 3D image stack
    Args:
        stack (4D ndarray): 1xZxHxW image array
    """

    def __init__(self, stack, masks={}, supervised=True, transform=None,
                 max_size=None, mask_mips={}, dataset_mip=None):
        self.supervised = supervised
        self.stack = stack
        self.N = self.stack.shape[1] * self.stack.shape[0]
        if not self.supervised:
            self.N -= 1
        if max_size:
            self.N = min(self.N, max_size)
        self.transform = transform
        self.masks = masks
        self.mask_mips = mask_mips
        self.dataset_mip = dataset_mip

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        # 2*(stack.shape[1]-1) consecutive image pairs
        return self.N

    def __getitem__(self, i):
        src = self.stack[0, i, :, :]
        if self.supervised:
            tgt = copy.copy(src)
        else:
            tgt = self.stack[0, i+1, :, :]

        #res = np.zeros((src.shape[0], src.shape[1], 2))
        res = torch.zeros((src.shape[0], src.shape[1], 2), device='cuda')
        masks = []

        for mask_type in mask_types:
            if mask_type in self.masks:
                src_mask = self.masks[mask_type][0, i, :, :]
                tgt_mask = self.masks[mask_type][0, i + 1, :, :]
            else:
                src_mask = np.zeros_like(src)
                tgt_mask = np.zeros_like(tgt)

            if self.dataset_mip is not None and mask_type in self.mask_mips:
                mask_mip = self.mask_mips[mask_type]
                if self.dataset_mip > mask_mip:
                    raise Exception("Not implemented")

                for _ in range(self.dataset_mip, mask_mip):
                    src_mask = resize(src_mask, [src_mask.shape[0] * 2,
                                                 src_mask.shape[1] * 2])
                    tgt_mask = resize(tgt_mask, [tgt_mask.shape[0] * 2,
                                                 tgt_mask.shape[1] * 2])
            masks.append(src_mask)
            masks.append(tgt_mask)

        sample = torch.cat([torch.cuda.FloatTensor(src).unsqueeze(0),
                            torch.cuda.FloatTensor(tgt).unsqueeze(0),
                            torch.cuda.FloatTensor(res).permute(2, 0, 1)] +
                           [torch.cuda.FloatTensor(m).unsqueeze(0) for m in masks], 0)

        return sample

def sample_to_bundle(sample):
    src = sample[0, 0]
    tgt = sample[0, 1]
    res = sample[0, 2:4].permute(1, 2, 0)

    masks = [sample[0, i] for i in range(4, sample.shape[1])]
    bundle = ((src, tgt, res), masks)
    return bundle

