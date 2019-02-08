import random
import math
import h5py
import torch
from torch.utils.data import Dataset, ConcatDataset

from aug import aug_input, rotate_and_scale, random_translation
from utilities.helpers import (upsample, downsample, gridsample_residual, 
                               dotdict)


def compile_dataset(*h5_paths, transform=None, num_samples=None):
    datasets = []
    for h5_path in h5_paths:
        h5f = h5py.File(h5_path, 'r')
        ds = [StackDataset(v, transform=transform, num_samples=num_samples) for v in h5f.values()]
        datasets.extend(ds)
    return ConcatDataset(datasets)


class StackDataset(Dataset):
    """Deliver consecutive image pairs from 3D image stack

    Args:
        stack (4D ndarray): 1xZxHxW image array
    """

    def __init__(self, stack, transform=None, num_samples=None):
        self.stack = stack
        self.N = (num_samples
                  if num_samples and num_samples < len(stack) else len(stack))
        self.transform = transform

    def __len__(self):
        return 2*len(self.stack)

    def __getitem__(self, id):
        X = self.stack[id % self.N].copy()  # prevent modifying the dataset
        if id % 2*self.N >= self.N:  # flip source and target
            # match i -> i+1 if id < N, else match i+1 -> i
            s, t, sc, tc, sf, tf = X.copy()
            X[0:6] = t, s, tc, sc, tf, sf
        if self.transform:
            X = self.transform(X)
        return X, id


############################################################################
# Data Loading Transforms
############################################################################


class Transform(object):
    """Superclass for tensor transforms
    """

    def only_if(self, condition):
        """Disable the Transform if the condition is false
        """
        if not condition:
            self.__call__ = lambda x: x


class ToFloatTensor(Transform):
    """Convert ndarray to FloatTensor
    """

    def __call__(self, X):
        return torch.from_numpy(X).to(torch.float)


class Preprocess(Transform):
    """Preprocess the input images to standardize contrast
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def __call__(self, X):
        if self.preprocessor is not None:
            X[..., 0:2, :, :] = self.preprocessor(X[..., 0:2, :, :])
        return X


class RandomRotateAndScale(Transform):
    """Randomly rotate & scale src and tgt images
    """

    def __call__(self, X):
        if random.randint(0, 20) != 0:
            X, _ = rotate_and_scale(X, None)
        return X.squeeze()


class RandomFlip(Transform):
    """Randomly flip src & tgt images
    """

    def __call__(self, X):
        if random.choice([True, False]):
            X = X.flip(len(X.shape) - 1)
        if random.choice([True, False]):
            X = X.flip(len(X.shape) - 2)
        return X


class Split(Transform):
    """Split sample into a (src, tgt) pair
    """

    def __call__(self, X):
        return dotdict({
            'src': {
                'image': X[0:1],
                'fold_mask': X[4:5],
            },
            'tgt': {
                'image': X[1:2],
                'fold_mask': X[4:5],
            },
        })


class RandomTranslation(Transform):
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


class RandomAugmentation(Transform):
    """Apply random Gaussian noise, cutouts, & brightness adjustment
    """

    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, X):
        src, tgt = X.src.image.clone(), X.tgt.image.clone()
        src_aug, src_aug_masks = aug_input(src)
        tgt_aug, tgt_aug_masks = aug_input(tgt)
        X.src.aug = src_aug
        X.tgt.aug = tgt_aug
        X.src.aug_masks = src_aug_masks
        X.tgt.aug_masks = tgt_aug_masks
        return X


class RandomField(Transform):
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
        shape = X.src.image.shape
        zero = torch.zeros(shape, device='cuda')
        zero = torch.cat([zero, zero.clone()], 1)
        smaller = downsample(self.num_downsamples)(zero)
        std = self.max_displacement / shape[-2] / math.sqrt(2)
        field = torch.nn.init.normal_(smaller, mean=0, std=std)
        field = upsample(self.num_downsamples)(field)
        X.truth = field.permute(0, 2, 3, 1)
        for k in X.tgt:
            X.tgt[k] = gridsample_residual(
                X.src[k], X.truth, padding_mode='zeros')
        return X


class ToDevice(Transform):
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
        return X
