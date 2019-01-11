import torch
import torch.nn as nn
import numpy as np
from utilities.helpers import save_chunk, compose_functions
from scipy.ndimage import gaussian_filter
from skimage.filters import rank
from skimage.morphology import disk


class Preprocessor:
    """
    Code to preprocess the data.
    This can include contrast normalization, masking, etc.

    While this does necessarily not need to be a PyTorch module, it inherits
    from nn.Module to make it easier to parallelize with DataParallel if
    desired.
    At the very least, it must be a callable.
    """

    def __init__(self, mip=2, f=None, *args, **kwargs):
        self.mip = min(5, mip)
        self.f = compose_functions([self.rescale, self.contrast]) if f is None else f

    def old_contrast(self, t, l=145, h=210):
        zeromask = (t == 0)
        t[t < l] = l
        t[t > h] = h
        t *= 255.0 / (h-l+1)
        t = (t - np.min(t) + 1) / 255.
        t[zeromask] = 0
        return t

    def rescale(self, img, factor=1., dtype=np.float32, squeeze_epsilon=1/255.):
        zm = img == 0
        img = img.astype(np.float32)
        if np.max(img) > np.min(img):
            unit = (img - np.min(img)) / (np.max(img) - np.min(img))
            unit_eps = unit * (1-squeeze_epsilon) + squeeze_epsilon
            scaled = unit * factor
            output = scaled.astype(dtype)
        else:
            output = np.zeros(img.shape)
        assert np.min(output) >= 0
        assert np.max(output) <= factor
        output[zm] = 0
        return output

    def masked_gaussian_filter(self, img, r, mask):
        pre = img[~mask]
        img[~mask] = np.mean(img[mask])
        filtered = gaussian_filter(img, r)
        filtered[~mask] = 0
        img[~mask] = pre
        return filtered

    def highpass(self, img, radius=18, radius_func=lambda m, r: r // (m+1)):
        zm = img == 0
        r = radius_func(self.mip, radius)
        smoothed = self.masked_gaussian_filter(img, r, (img!=0))
        filtered = img - smoothed
        filtered[zm] = 0
        return self.rescale(filtered)

    def contrast(self, img, radius=128, radius_func=lambda m, r: r // (m+1)):
        rescaled = self.rescale(img, factor=255., dtype=np.uint8)
        r = radius_func(self.mip, radius)
        equalized = rank.equalize(rescaled, disk(r), mask=(rescaled!=0))
        return self.rescale(equalized)

    def apply_slice(self, img):
        return self.f(img).astype(np.float32)

    def apply_stack(self, img):
        slice_results = [np.expand_dims(self.apply_slice(img[0,i]), 0) for i in range(img.shape[1])]
        stacked = np.expand_dims(np.concatenate(slice_results), 0)
        return stacked

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = img.cpu().numpy()
        if img.ndim == 2:
            return torch.from_numpy(self.apply_slice(img))
        elif img.ndim == 3:
            return torch.from_numpy(self.apply_stack(img[np.newaxis, ...]))[0]
        else:
            return torch.from_numpy(self.apply_stack(img))
