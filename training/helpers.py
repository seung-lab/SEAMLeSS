import os
import shutil
from pathlib import Path
from moviepy.editor import ImageSequenceClip
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.transform import rescale
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

def compose_functions(fseq):
    def compose(f1, f2):
        return lambda x: f2(f1(x))
    return reduce(compose, fseq, lambda _: _)


def copy(src, dst):
    """
    A wrapper for the shutil copy function, but that accepts path objects.
    The shutil library will be updated to accept them directly in a later
    version of python, and so this will no longer be needed, but for now,
    this seemed cleaner than having explicit conversions everywere.
    """
    if isinstance(src, Path):
        src = str(src)
    if isinstance(dst, Path):
        dst = str(dst)
    shutil.copy(src, dst)


def copy_state_to_model(archive_params, model):
    size_map = [
        'number of out channels',
        'number of in channels',
        'kernel x dimension',
        'kernel y dimension'
    ]

    model_params = dict(model.named_parameters())
    model_keys = sorted(model_params.keys())
    archive_keys = sorted([k for k in archive_params.keys() if 'seq' not in k and 'pelist' not in k])
    assert len(archive_keys) <= len(model_keys), 'Cannot load archive with more parameters than model ({}, {}).'.format(len(archive_keys), len(model_keys))

    approx = 0
    skipped = 0
    new = len(set(model_keys)) - len(set(archive_keys))
    for key in archive_keys:
        if key not in model_keys:
            print('[WARNING]   Key ' + key + ' present in archive but not in model; skipping.')
            skipped += 1
            continue

        min_size = [min(mdim,adim) for mdim, adim in zip(list(model_params[key].size()), list(archive_params[key].size()))]
        msize, asize = model_params[key].size(), archive_params[key].size()
        if msize != asize:
            approx += 1
            wrong_dim = -1
            for dim in range(len(msize)):
                if msize[dim] != asize[dim]:
                    wrong_dim = dim
                    break
            print('[WARNING]   ' + key + ' has different ' + size_map[wrong_dim] + ' in model and archive: ' + str(model_params[key].size()) + ', ' + str(archive_params[key].size()))
            varchive = torch.std(archive_params[key])
            vmodel = torch.std(model_params[key])
            model_params[key].data -= torch.mean(model_params[key].data)
            model_params[key].data *= ((varchive / 5) / vmodel).data[0]
            model_params[key].data += torch.mean(archive_params[key])

        min_size_slices = tuple([slice(*(s,)) for s in min_size])
        model_params[key].data[min_size_slices] = archive_params[key][min_size_slices]

    print('Copied {} parameters exactly, {} parameters partially.'.format(len(model_keys) - approx - new, approx))
    print('Skipped {} parameters in archive, found {} new parameters in model.'.format(skipped, new))

def get_colors(angles, f, c):
    colors = f(angles)
    colors = c(colors)
    return colors

def dv(vfield, name=None, downsample=0.5):
    dim = vfield.shape[-2]
    assert type(vfield) == np.ndarray

    lengths = np.squeeze(np.sqrt(vfield[:,:,:,0] ** 2 + vfield[:,:,:,1] ** 2))
    lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
    angles = np.squeeze(np.angle(vfield[:,:,:,0] + vfield[:,:,:,1]*1j))

    angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles)) * np.pi
    angles -= np.pi/8
    angles[angles<0] += np.pi
    off_angles = angles + np.pi/4
    off_angles[off_angles>np.pi] -= np.pi

    scolors = get_colors(angles, f=lambda x: np.sin(x) ** 1.4, c=cm.viridis)
    ccolors = get_colors(off_angles, f=lambda x: np.sin(x) ** 1.4, c=cm.magma)

    # mix
    scolors[:,:,0] = ccolors[:,:,0]
    scolors[:,:,1] = (ccolors[:,:,1] + scolors[:,:,1]) / 2
    scolors = scolors[:,:,:-1] #
    scolors = 1 - (1 - scolors) * lengths.reshape((dim, dim, 1)) ** .8 #

    img = np_upsample(scolors, downsample) if downsample is not None else scolors

    if name is not None:
        plt.imsave(name + '.png', img)
    else:
        return img

def np_upsample(img, factor):
    if factor == 1:
        return img

    if img.ndim == 2:
        return rescale(img, factor)
    elif img.ndim == 3:
        b = np.empty((int(img.shape[0] * factor), int(img.shape[1] * factor), img.shape[2]))
        for idx in range(img.shape[2]):
            b[:,:,idx] = np_upsample(img[:,:,idx], factor)
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
        vfield[:,:,:,0] = vfield[:,:,:,0] - np.mean(vfield[:,:,:,0])
        vfield[:,:,:,1] = vfield[:,:,:,1] - np.mean(vfield[:,:,:,1])
        field[idx] = vfield
    return field[0] if wrap else field

def display_v(vfield, name=None, center=False):
    if center:
        center_field(vfield)

    if type(vfield) == list:
        dim = max([vf.shape[-2] for vf in vfield])
        vlist = [np.expand_dims(np_upsample(vf[0], dim/vf.shape[-2]), axis=0) for vf in vfield]
        for idx, _ in enumerate(vlist[1:]):
            vlist[idx+1] += vlist[idx]
        imgs = [dv(vf) for vf in vlist]
        gif(name, np.stack(imgs) * 255)
    else:
        assert (name is not None)
        dv(vfield, name)

def dvl(V_pred, name, mag=10):
    factor = V_pred.shape[1] // 100
    if factor > 1:
        V_pred = V_pred[:,::factor,::factor,:]
    V_pred *= 10
    plt.figure(figsize=(6,6))
    X, Y = np.meshgrid(np.arange(-1, 1, 2.0/V_pred.shape[-2]), np.arange(-1, 1, 2.0/V_pred.shape[-2]))
    U, V = np.squeeze(np.vsplit(np.swapaxes(V_pred,0,-1),2))
    colors = np.arctan2(U,V)   # true angle
    plt.title('V_pred')
    plt.gca().invert_yaxis()
    Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002, angles='uv', pivot='tail')
    qk = plt.quiverkey(Q, 10.0, 10.0, 2, r'$2 \frac{m}{s}$', labelpos='E', \
                       coordinates='figure')

    plt.savefig(name + '.png')
    plt.clf()

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
    return f([center(var, (-2,-1), var.size()[-1] - size) for var in seq], 1)

def center(var, dims, d):
    if not isinstance(d, collections.Sequence):
        d = [d for i in range(len(dims))]
    for idx, dim in enumerate(dims):
        if d[idx] == 0:
            continue
        var = var.narrow(dim, d[idx]/2, var.size()[dim] - d[idx])
    return var

def crop(data_2d, crop):
    return data_2d[crop:-crop,crop:-crop]

def save_chunk(chunk, name, norm=True):
    if type(chunk) != np.ndarray:
        chunk = chunk.cpu().numpy()
    chunk = np.squeeze(chunk).astype(np.float64)
    if norm:
        chunk[:50,:50] = 0
        chunk[:10,:10] = 1
        chunk[-50:,-50:] = 1
        chunk[-10:,-10:] = 0
    plt.imsave(name + '.png', 1 - chunk, cmap='Greys')

def gif(filename, array, fps=2, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array *= 255
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # add 'signature' block to top left and bottom right
    if array.shape[1] > 1000:
        array[:,:50,:50] = 0
        array[:,:10,:10] = 255
        array[:,-50:,-50:] = 255
        array[:,-10:,-10:] = 0

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps, verbose=False)
    return clip


def downsample(x=1, type='average'):
    if x > 0:
        if type == 'average':
            return nn.AvgPool2d(2**x, count_include_pad=False)
        elif type == 'max':
            return nn.MaxPool2d(2**x)
        else:
            raise ValueError('Unrecognized pooling type: {}'.format(type))
    else:
        return (lambda y: y)


def upsample(x=1):
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
    `source` arranged as a PyTorch image, and `field` as a PyTorch vector field.

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
    return F.grid_sample(source, scaled_field, mode="bilinear", padding_mode=padding_mode)

def gridsample_residual(source, residual, padding_mode):
    """
    Similar to `gridsample()`, but takes a residual field.
    This abstracts away generation of the appropriate identity grid.
    """
    field = residual + identity_grid(residual.shape, device=residual.device)
    return gridsample(source, field, padding_mode)

def _create_identity_grid(size):
    with torch.no_grad():
        id_theta = torch.Tensor([[[1,0,0],[0,1,0]]]) # identity affine transform
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
    if isinstance(size,torch.Size):
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


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, store=False):
        self.reset(store)

    def reset(self, store=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = None
        if store:
            self.history = []
        self.warned = False

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            if not self.warned:
                warnings.warn('Accumulating a pytorch tensor can cause a gpu '
                              'memory leak. Converting to a python scalar.')
                self.warned = True
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.history is not None:
            self.history += [val]*n
