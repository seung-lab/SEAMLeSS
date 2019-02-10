import os
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import rescale
from functools import reduce


def compose_functions(fseq):
    def compose(f1, f2):
        return lambda x: f2(f1(x))
    return reduce(compose, fseq, lambda _: _)


def np_upsample(img, factor):
    if factor == 1:
        return img

    if img.ndim == 2:
        return rescale(img, factor)
    elif img.ndim == 3:
        b = np.empty((int(img.shape[0] * factor),
                      int(img.shape[1] * factor), img.shape[2]))
        for idx in range(img.shape[2]):
            b[:, :, idx] = np_upsample(img[:, :, idx], factor)
        return b
    else:
        assert False


def np_downsample(img, factor):
    data_4d = np.expand_dims(img, axis=1)
    result = nn.AvgPool2d(factor)(torch.from_numpy(data_4d))
    return result.numpy()[:, 0, :, :]


def center_field(field):
    wrap = type(field) == np.ndarray
    if wrap:
        field = [field]
    for idx, vfield in enumerate(field):
        vfield[:, :, :, 0] = vfield[:,:,:,0] - np.mean(vfield[:,:,:,0])
        vfield[:, :, :, 1] = vfield[:,:,:,1] - np.mean(vfield[:,:,:,1])
        field[idx] = vfield
    return field[0] if wrap else field


def reverse_dim(var, dim):
    if var is None:
        return var
    idx = range(var.size()[dim] - 1, -1, -1)
    idx = torch.LongTensor(idx)
    if var.is_cuda:
        idx = idx.cuda()
    return var.index_select(dim, idx)


def reduce_seq(seq, f):
    size = min([x.size()[-1] for x in seq])
    return f([center(var, (-2, -1),
              var.size()[-1] - size) for var in seq], 1)


def center(var, dims, d):
    if not isinstance(d, collections.Sequence):
        d = [d for i in range(len(dims))]
    for idx, dim in enumerate(dims):
        if d[idx] == 0:
            continue
        var = var.narrow(dim, d[idx]/2, var.size()[dim] - d[idx])
    return var


def crop(data_2d, crop):
    return data_2d[crop:-crop, crop:-crop]


def downsample(x):
    if x > 0:
        return nn.AvgPool2d(2**x, count_include_pad=False)
    else:
        return (lambda y: y)


def upsample(x):
    if x > 0:
        return nn.Upsample(scale_factor=2**x, mode='bilinear')
    else:
        return (lambda y: y)


def gridsample(source, field, padding_mode):
    """
    A version of the PyTorch grid sampler that uses size-agnostic conventions.
    Vectors with values -1 or +1 point to the actual edges of the images
    (as opposed to the centers of the border pixels as in PyTorch 4.1).
    `source` and `field` should be PyTorch tensors on the same GPU, with
    `source` arranged as a PyTorch image, and `field` as a PyTorch vector
    field.
    `padding_mode` is required because it is a significant consideration.
    It determines the value sampled when a vector is outside the range [-1,1]
    Options are:
     - "zero" : produce the value zero (okay for sampling images with zero as
                background, but potentially problematic for sampling masks and
                terrible for sampling from other vector fields)
     - "border" : produces the value at the nearest inbounds pixel (great for
                  masks and residual fields)
    If sampling a field (ie. `source` is a vector field), best practice is to
    subtract out the identity field from `source` first (if present) to get a
    residual field.
    Then sample it with `padding_mode = "border"`.
    This should behave as if source was extended as a uniform vector field
    beyond each of its boundaries.
    Note that to sample from a field, the source field must be rearranged to
    fit the conventions for image dimensions in PyTorch. This can be done by
    calling `source.permute(0,3,1,2)` before passing to `gridsample()` and
    `result.permute(0,2,3,1)` to restore the result.
    """
    if source.shape[2] != source.shape[3]:
        raise NotImplementedError('Grid sampling from non-square tensors '
                                  'not yet implementd here.')
    scaled_field = field * source.shape[2] / (source.shape[2] - 1)
    return F.grid_sample(source, scaled_field, mode="bilinear",
                         padding_mode=padding_mode)


def gridsample_residual_2d(source, residual, padding_mode):
    """
    Similar to `gridsample()`, but takes a residual field.
    This abstracts away generation of the appropriate identity grid.
    """
    source = torch.FloatTensor(source).unsqueeze(0).unsqueeze(0)
    residual = torch.FloatTensor(residual).unsqueeze(0)
    return gridsample_residual(source, residual, padding_mode)


def gridsample_residual(source, residual, padding_mode):
    """
    Similar to `gridsample()`, but takes a residual field.
    This abstracts away generation of the appropriate identity grid.
    """
    field = residual + identity_grid(residual.shape, device=residual.device)
    return gridsample(source, field, padding_mode)


def _create_identity_grid(size):
    with torch.no_grad():
        id_theta = torch.cuda.FloatTensor([[[1,0,0],[0,1,0]]]) # identity affine transform
        I = F.affine_grid(id_theta,torch.Size((1,1,size,size)))
        I *= (size - 1) / size # rescale the identity provided by PyTorch
        return I


def identity_grid(size, cache=False, device=None):
    """
    Returns a size-agnostic identity field with -1 and +1 pointing to the
    corners of the image (not the centers of the border pixels as in
    PyTorch 4.1).
    Use `cache = True` to cache the identity for faster recall.
    This can speed up recall, but may be a burden on cpu/gpu memory.
    `size` can be either an `int` or a `torch.Size` of the form
    `(N, C, H, W)`. `H` and `W` must be the same (a square tensor).
    `N` and `C` are ignored.
    """
    if isinstance(size, torch.Size):
        if (size[2] == size[3] # image
            or (size[3] == 2 and size[1] == size[2])): # field
            size = size[2]
        else:
            raise ValueError("Bad size: {}. Expected a square tensor size.".format(size))
    if device is None:
        device = torch.cuda.current_device()
    if size in identity_grid._identities:
        return identity_grid._identities[size].to(device)
    I = _create_identity_grid(size)
    if cache:
        identity_grid._identities[size] = I
    return I.to(device)
identity_grid._identities = {}
