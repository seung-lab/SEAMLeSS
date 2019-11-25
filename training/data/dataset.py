import math
import random
from itertools import permutations

import h5py
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import ConcatDataset, Dataset

from aug import aug_input, random_translation, rotate_and_scale
from utilities.helpers import dotdict, downsample


def compile(*h5_paths, transform=None, num_samples=None, repeats=1):
    datasets = []
    for h5_path in h5_paths:
        ds = [
            StackDataset(
                h5_path, transform=transform, num_samples=num_samples, repeats=repeats
            )
        ]
        datasets.extend(ds)
    return ConcatDataset(datasets)


class StackDataset(Dataset):
    """Deliver image pairs from 4D image stack

    Args:
        h5f: HDF5 file with two datasets, "images" & "masks"
             both datasets organized as 4D ndarrays,
             Sx2xHxW image array (S is no. of sample pairs)
    """

    def __init__(self, h5_path, transform=None, num_samples=None, repeats=1):
        self.h5_path = h5_path
        self.transform = transform
        self.repeats = repeats
        self.images = None
        self.masks = None

        with h5py.File(self.h5_path, "r") as h5f:
            assert "images" in h5f.keys()
            if "masks" in h5f.keys():
                assert h5f["masks"].shape[-4:-2] == h5f["images"].shape[-4:-2]
            self.N = (
                num_samples
                if num_samples and num_samples < len(h5f["images"])
                else len(h5f["images"])
            )
            self.stack_height = h5f["images"].shape[-3]

        self.permutations = tuple(permutations(range(self.stack_height), 2))

    def __len__(self):
        return self.N * self.repeats * len(self.permutations)

    def __getitem__(self, id):
        if self.images is None:
            h5f = h5py.File(self.h5_path, "r")
            self.images = h5f["images"]
            if "masks" in h5f.keys():
                self.masks = h5f["masks"]
            else:
                self.masks = np.zeros_like(self.images)

        img_id = id % self.N
        perm_id = id % len(self.permutations)
        s, t = self.images[img_id][self.permutations[perm_id], ...]
        sm, tm = self.masks[img_id][self.permutations[perm_id], ...]

        X = np.zeros_like(s, shape=(4, s.shape[-2], s.shape[-1]))
        if sm.shape[0] != s.shape[0]:
            sm = resize(sm, s.shape)
            tm = resize(tm, t.shape)

        X[0:4] = s, t, sm, tm
        if self.transform:
            X = self.transform(X)
        return X, id


############################################################################
# Data Loading Transforms
############################################################################


class OnlyIf(object):
    """Wrapper transform that applies the underlying only if a
    condition is true
    """

    def __init__(self, transform, condition):
        self.transform = transform
        self.condition = condition

    def __call__(self, X):
        if not self.condition:
            return X
        return self.transform(X)


class ToFloatTensor(object):
    """Convert ndarray to FloatTensor
    """

    def __call__(self, X):
        return torch.from_numpy(X).to(torch.float)


class Preprocess(object):
    """Preprocess the input images to standardize contrast
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def __call__(self, X):
        if self.preprocessor is not None:
            X[..., 0:2, :, :] = self.preprocessor(X[..., 0:2, :, :])
        return X


class RandomRotateAndScale(object):
    """Randomly rotate & scale src and tgt images
    """

    def __call__(self, X):
        if random.randint(0, 20) != 0:
            X, _ = rotate_and_scale(X, None)
        return X.squeeze()


class RandomFlip(object):
    """Randomly flip src & tgt images
    """

    def __call__(self, X):
        if random.choice([True, False]):
            X = X.flip(len(X.shape) - 1)
        if random.choice([True, False]):
            X = X.flip(len(X.shape) - 2)
        return X


class Split(object):
    """Split sample into a (src, tgt) pair
    """

    def __call__(self, X):
        return dotdict({
            'src': {
                'image': X[0:1],
                'mask': X[2:3],
            },
            'tgt': {
                'image': X[1:2],
                'mask': X[3:4], 
            },
        })


class RandomTranslation(object):
    """Randomly translate src & tgt images separately
    """

    def __init__(self, max_displacement=2**6):
        self.max_displacement = max_displacement

    def __call__(self, X):
        if random.randint(0, 5) != 0:
            offset = None
            for k in X.src:
                X.src[k], offset = random_translation(
                    X.src[k], self.max_displacement, offset)
        if random.randint(0, 5) != 0:
            offset = None
            for k in X.tgt:
                X.tgt[k], offset = random_translation(
                    X.tgt[k], self.max_displacement, offset)
        return X


class RandomField(object):
    """Genenerates a random vector field smoothed by bilinear interpolation.

    The vectors generated will have values representing displacements of
    between (approximately) `-max_displacement` and `max_displacement` pixels.
    The actual values, however, will be scaled to the spatial transformer
    standard, where -1 and 1 represent the edges of the image.

    `num_downsamples` dictates the block size for the random field.
    Each block will have size `2**num_downsamples`.
    """

    def __init__(self, max_displacement=2, num_downsamples=7):
        self.max_displacement = max_displacement
        self.num_downsamples = num_downsamples

    def __call__(self, X):
        zero = torch.zeros_like(X.src.image).unsqueeze(0)
        smaller = downsample(self.num_downsamples)(zero)
        std = self.max_displacement / zero.shape[-2] / math.sqrt(2)
        smaller = torch.cat([smaller, smaller.clone()], 1)
        field = torch.nn.init.normal_(smaller, mean=0, std=std).field_()
        field = field.up(self.num_downsamples)
        X.truth = field.squeeze()
        for k in X.tgt:
            X.tgt[k] = field.sample(X.src[k])
        return X


class RandomAugmentation(object):
    """Apply random Gaussian noise, cutouts, & brightness adjustment
    """

    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, X):
        src, tgt = X.src.image.clone(), X.tgt.image.clone()
        src_aug, src_aug_masks = aug_input(src.squeeze(0))
        tgt_aug, tgt_aug_masks = aug_input(tgt.squeeze(0))
        X.src.aug = src_aug.unsqueeze(0)
        X.tgt.aug = tgt_aug.unsqueeze(0)
        X.src.aug_masks = src_aug_masks
        X.tgt.aug_masks = tgt_aug_masks
        return X


class ToDevice(object):
    """Move tensors to a specific device
    """

    def __init__(self, device=0):
        self.device = device

    def __call__(self, X):
        for k, v in X.items():
            if isinstance(v, torch.Tensor):
                X[k] = v.to(device=self.device)
            elif isinstance(v, dict):
                X[k] = self(v)
            elif isinstance(v, list):
                X[k] = [x.to(device=self.device) for x in v]
        return X
