import os
import sys
import shutil
import warnings
import math
from pathlib import Path
from moviepy.editor import ImageSequenceClip
import numpy as np
import collections
import numbers
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul, pow, mul, reciprocal
from torch.nn.functional import interpolate
from torch.nn import AvgPool2d, LPPool2d
from torch.nn.functional import softmax
from itertools import combinations
from skimage.transform import rescale
from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum
from functools import reduce
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: 402
plt.switch_backend('agg')
import matplotlib.cm as cm  # noqa: 402


def compose_functions(fseq):
    def compose(f1, f2):
        return lambda x: f2(f1(x))
    return reduce(compose, fseq, lambda _: _)


def cp(src, dst):
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


@torch.no_grad()
def load_model_from_dict(model, archive_params):
    model_params = model.state_dict(keep_vars=True)
    model_keys = sorted(model_params.keys())
    archive_keys = sorted(archive_params.keys())

    dropped = 0
    approx = 0
    for key in archive_keys:
        if key not in model_keys:
            print('[WARNING]   Key {} present in archive but not in model; '
                  .format(key))
            dropped += 1
            continue
        if model_params[key].shape != archive_params[key].shape:
            print('[WARNING]   {} has different shape in model and archive: '
                  '{}, {}'.format(key, model_params[key].shape,
                                  archive_params[key].shape))
            min_slices = tuple(slice(min(mdim, adim)) for mdim, adim
                               in zip(model_params[key].shape,
                                      archive_params[key].shape))
            model_params[key].data[min_slices] = (
                archive_params[key][min_slices])
            model_params[key].data = (
                (model_params[key] - model_params[key].mean())
                / model_params[key].std())
            model_params[key].data = (
                (model_params[key] * archive_params[key].std())
                + archive_params[key].mean())
            approx += 1
            continue
        model_params[key].data = archive_params[key]
    new = 0
    for key in model_keys:
        if key not in archive_keys:
            print('[WARNING]   Key {} present in model but not in archive; '
                  .format(key))
            new += 1
    print('Copied {} parameters exactly, {} parameters partially.'
          .format(len(archive_keys) - dropped - approx, approx))
    print('Skipped {} parameters in archive, found {} new parameters in model.'
          .format(dropped, new))


def get_colors(angles, f, c):
    colors = f(angles)
    colors = c(colors)
    return colors


def dv(vfield, name=None, downsample=0.5):
    dim = vfield.shape[-2]
    assert type(vfield) == np.ndarray

    lengths = np.squeeze(np.sqrt(vfield[:, :, :, 0] ** 2
                                 + vfield[:, :, :, 1] ** 2))
    lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
    angles = np.squeeze(np.angle(vfield[:, :, :, 0] + vfield[:, :, :, 1]*1j))

    angles = ((angles - np.min(angles))
              / (np.max(angles) - np.min(angles)) * np.pi)
    angles -= np.pi/8
    angles[angles < 0] += np.pi
    off_angles = angles + np.pi/4
    off_angles[off_angles > np.pi] -= np.pi

    scolors = get_colors(angles, f=lambda x: np.sin(x) ** 1.4, c=cm.viridis)
    ccolors = get_colors(off_angles, f=lambda x: np.sin(x) ** 1.4, c=cm.magma)

    # mix
    scolors[:, :, 0] = ccolors[:, :, 0]
    scolors[:, :, 1] = (ccolors[:, :, 1] + scolors[:, :, 1]) / 2
    scolors = scolors[:, :, :-1]
    scolors = 1 - (1 - scolors) * lengths.reshape((dim, dim, 1)) ** .8

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
        vfield[:, :, :, 0] = vfield[:, :, :, 0] - np.mean(vfield[:, :, :, 0])
        vfield[:, :, :, 1] = vfield[:, :, :, 1] - np.mean(vfield[:, :, :, 1])
        field[idx] = vfield
    return field[0] if wrap else field


def display_v(vfield, name=None, center=False):
    if center:
        center_field(vfield)

    if type(vfield) == list:
        dim = max([vf.shape[-2] for vf in vfield])
        vlist = [np.expand_dims(np_upsample(vf[0], dim/vf.shape[-2]), axis=0)
                 for vf in vfield]
        for idx, _ in enumerate(vlist[1:]):
            vlist[idx+1] += vlist[idx]
        imgs = [dv(vf) for vf in vlist]
        gif(name, np.stack(imgs) * 255)
    else:
        assert (name is not None)
        dv(vfield, name)


def dvl(V_pred, name, mag=100):
    factor = V_pred.shape[1] // 100
    if factor > 1:
        # subsample the field
        V_pred = V_pred.unfold(1, factor, factor)[..., 0]
        V_pred = V_pred.unfold(2, factor, factor)[..., 0]
    V_pred = V_pred * mag
    if isinstance(V_pred, torch.Tensor):
        V_pred = V_pred.cpu().numpy()
    plt.figure(figsize=(6, 6))
    X, Y = np.meshgrid(np.arange(-1, 1, 2.0/V_pred.shape[-2]),
                       np.arange(-1, 1, 2.0/V_pred.shape[-2]))
    U, V = np.squeeze(np.vsplit(np.swapaxes(V_pred, 0, -1), 2))
    colors = np.arctan2(U, V)  # true angle
    plt.title(Path(name).stem)
    plt.gca().invert_yaxis()
    Q = plt.quiver(X, Y, U, V, colors, scale=6, width=0.002,
                   angles='uv', pivot='tail')
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
    return f([center(var, (-2, -1), var.size()[-1] - size) for var in seq], 1)


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


def save_chunk(chunk, name, norm=True):
    if type(chunk) != np.ndarray:
        chunk = chunk.cpu().numpy()
    chunk = np.squeeze(chunk).astype(np.float64)
    if norm:
        chunk[:50, :50] = 0
        chunk[:10, :10] = 1
        chunk[-50:, -50:] = 1
        chunk[-10:, -10:] = 0
    plt.imsave(name + '.png', 1 - chunk, cmap='Greys')


def gif(filename, array, fps=2, scale=1.0, norm=False):
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
    tqdm.pos = 0  # workaround for tqdm bug when using it in multithreading

    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array *= 255
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # add 'signature' block to top left and bottom right
    if norm and array.shape[1] > 1000:
        array[:, :50, :50] = 0
        array[:, :10, :10] = 255
        array[:, -50:, -50:] = 255
        array[:, -10:, -10:] = 0

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


def gridsample_residual(source, residual, padding_mode):
    """
    DEPRECATED: This function has been renamed and is deprecated in favor of
    `grid_sample()`

    A version of the PyTorch grid sampler that uses size-agnostic residual
    conventions.

    The vector field encodes relative displacements from which to pull from the
    image, where vectors with values -1 or +1 reference a displacement equal
    to the distance from the center point to the actual edges of the images
    (as opposed to the centers of the border pixels as in PyTorch 1.0).

    Args:
        `input` and `field` (Tensors): should be PyTorch tensors on the same
            GPU or CPU, with `input` arranged as a PyTorch image
            :math:`(N, C, H_in, W_in)`, and `field` as a PyTorch vector field
            :math:`(N, H_out, W_out, 2)`.
            The shape of the output image will be `(N, C, H_out, W_out)`
        `padding_mode` (str): required because it is a significant
            consideration. It determines the value sampled when a vector's
            source falls outside of the input.
            Options are:
             - "zero" : produce the value zero (okay for sampling images with
                        zero as background, but potentially problematic for
                        sampling masks and terrible for sampling from other
                        vector fields)
             - "border" : produces the value at the nearest inbounds pixel
                          (great for masks and residual fields)

    Returns:
        `output` (Tensor): the input after being warped by `field`,
        having shape :math:`(N, C, H_out, W_out)`

    Note:
        If sampling a field (ie. `input` is itself a vector field), the input
        field must be rearranged to fit the conventions for image dimensions
        in PyTorch. This can be done by calling `input.permute(0,3,1,2)`
        before passing to `gridsample()` and `result.permute(0,2,3,1)` to
        restore the result.
    """
    print('gridsample_residual() has been renamed and is deprecated in favor '
          'of grid_sample()', file=sys.stderr)
    return grid_sample(source, residual, padding_mode)


def grid_sample(input, field, padding_mode, mode='bilinear'):
    """A version of the PyTorch grid sampler that uses size-agnostic residual
    conventions.

    The vector field encodes relative displacements from which to pull from the
    input, where vectors with values -1 or +1 reference a displacement equal
    to the distance from the center point to the actual edges of the input
    (as opposed to the centers of the border pixels as in PyTorch 1.0).

    Args:
        `input` and `field` (Tensors): should be PyTorch tensors on the same
            GPU or CPU, with `input` arranged as a PyTorch image
            :math:`(N, C, H_in, W_in)`, and `field` as a PyTorch vector field
            :math:`(N, H_out, W_out, 2)`.
            The shape of the output will be :math:`(N, C, H_out, W_out)`.
        `padding_mode` (str): required because it is a significant
            consideration. It determines the value sampled when a vector's
            source falls outside of the input.
            Options are:
             - "zero" : produce the value zero (okay for sampling images with
                        zero as background, but potentially problematic for
                        sampling masks and terrible for sampling from other
                        vector fields)
             - "border" : produces the value at the nearest inbounds pixel
                          (great for masks and residual fields)

    Returns:
        `output` (Tensor): the input after being warped by `field`,
        having shape :math:`(N, C, H_out, W_out)`

    Note:
        If sampling a field (ie. `input` is itself a vector field), the input
        field must be rearranged to fit the conventions for image dimensions
        in PyTorch. This can be done by calling `input.permute(0,3,1,2)`
        before passing to `gridsample()` and `result.permute(0,2,3,1)` to
        restore the result.
        This is simplified by calling `grid_sample_field()`
    """
    field = field + identity_grid(field.shape, device=field.device)
    if input.shape[2] != input.shape[3]:
        raise NotImplementedError('Grid sampling from non-square tensors '
                                  'not yet implementd here.')
    scaled_field = field * input.shape[2] / (input.shape[2] - 1)
    return F.grid_sample(input, scaled_field, mode=mode,
                         padding_mode=padding_mode)


def grid_sample_field(input_field, warp_field, padding_mode='border',
                      mode='bilinear'):
    """A version of the PyTorch grid sampler that uses size-agnostic residual
    conventions, and samples from another vector field.

    The vector field encodes relative displacements from which to pull from the
    input, where vectors with values -1 or +1 reference a displacement equal
    to the distance from the center point to the actual edges of the input
    (as opposed to the centers of the border pixels as in PyTorch 1.0).

    Args:
        `input_field` and `warp_field` (Tensors): should be PyTorch tensors on
            the same GPU or CPU, arranged as PyTorch vector fields with shapes
            :math:`(N, H_in, W_in, 2)` and :math:`(N, H_out, W_out, 2)`
            respectively.
            The shape of the output will be :math:`(N, H_out, W_out, 2)`.

    Returns:
        `output` (Tensor): the input field after being warped by the warp
        field, having shape :math:`(N, H_out, W_out, 2)`

    Note:
        If sampling a warp_field (ie. `input_field` is itself a vector field), the input
        field must be rearranged to fit the conventions for image dimensions
        in PyTorch. This can be done by calling `input.permute(0,3,1,2)`
        before passing to `gridsample()` and `result.permute(0,2,3,1)` to
        restore the result.
    """
    input = input_field.permute(0, 3, 1, 2)
    output = grid_sample(input, warp_field, padding_mode=padding_mode,
                         mode=mode)
    return output.permute(0, 2, 3, 1)


def compose_fields(f, g):
    r"""Compose two vector fields `f⚬g` such that `(f⚬g)(x) ~= f(g(x))`

    Returns:
        a vector field such that when it is used to grid sample,
        it is the (approximate) equivalent of sampling with `g`
        and then with `f`.

    The reason this is only an approximate equivalence is because when sampling
    twice, information is inevitably lost in the intermediate stage.
    Sampling with the composed field is therefore more precise.
    """
    return f + grid_sample_field(g, f)


@torch.no_grad()
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
    def _create_identity_grid(size, device):
        if 'cpu' in str(device):  # identity affine transform
            id_theta = torch.FloatTensor([[[1, 0, 0],
                                           [0, 1, 0]]])
        else:
            id_theta = torch.cuda.FloatTensor([[[1, 0, 0],
                                                [0, 1, 0]]], device=device)
        Id = F.affine_grid(id_theta, torch.Size((1, 1, size, size)))
        Id *= (size - 1) / size  # rescale the identity provided by PyTorch
        return Id

    if isinstance(size, torch.Size):
        batch_dim = size[0]
        if (size[2] == size[3]  # image
                or (size[3] == 2 and size[1] == size[2])):  # field
            size = size[2]
        else:
            raise ValueError("Bad size: {}. Expected a square tensor size."
                             .format(size))
    else:
        batch_dim = 1
    if device is None:
        device = torch.cuda.current_device()
    if size in identity_grid._identities:
        Id = identity_grid._identities[size].copy()
    else:
        Id = _create_identity_grid(size, device)
        if cache:
            identity_grid._identities[size] = Id.copy()
    if batch_dim > 1:
        Id = torch.cat([Id] * batch_dim)
    return Id.to(device)
identity_grid._identities = {}


def upsample_field(field, src_mip, dst_mip):
    """Upsample vector field from src_mip to dst_mip
    """
    field = field.permute(0, 3, 1, 2)
    upsampled_field = upsample(src_mip-dst_mip)(field)
    return upsampled_field.permute(0, 2, 3, 1)


def is_identity(field, eps=0.):
    """Check if field is the identity, up to some error `eps` (0 by default)
    """
    return torch.min(field) >= -eps and torch.max(field) <= eps


def get_affine_field(aff, offset, size, device):
    """Create a residual field for an affine transform within bbox

    Args:
        aff: 2x3 ndarray defining affine transform at MIP0
        offset: iterable with MIP0 offset
        size: either an `int` or a `torch.Size` of the form
            `(N, C, H, W)`. `H` and `W` must be the same (a square tensor).
            `N` and `C` are ignored.

    Returns:
        field torch tensor for affine field within bbox as MIP0 absolute
        residuals

    Note:
        the affine matrix defines the transformation that warps to destination
        to the source, such that,
        ```
        \vec{x_s} = A \vec{x_d}
        ```
        where x_s is a point in the source image, x_d a point in the
        destination image, and A is the affine matrix. The field returned
        will be defined over the destination image. So the matrix A should
        define the location in the source image that contribute to a pixel
        in the destination image.
    """
    A = torch.cuda.FloatTensor(np.concatenate([aff, [[0, 0, 1]]], axis=0),
                               device=device)
    B = torch.cuda.FloatTensor([[1., 0, offset[0]],
                                [0, 1., offset[1]],
                                [0, 0, 1]], device=device)
    Bi = torch.cuda.FloatTensor([[1., 0, -offset[0]],
                                 [0, 1., -offset[1]],
                                 [0, 0, 1]], device=device)
    theta = torch.mm(Bi, torch.mm(A, B))[:2].unsqueeze(0)
    print('get_affine_field \n{}'.format(theta.cpu().numpy()))
    M = F.affine_grid(theta, torch.Size((1, 1, size, size)))
    M *= (size - 1) / size  # rescale the grid provided by PyTorch
    return M - identity_grid(M.shape, device=M.device)


class dotdict(dict):
    """Allow accessing dict elements with dot notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)


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
        if isinstance(val, float) and not math.isfinite(val):
            return  # don't accumulate nan or inf
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.history is not None:
            self.history += [val]*n


def time_function(f, name=None, on=False):
    """
    Simple decorator used for timing functions.
    More capable timing suites exist, but this suffices for many purposes.

    Can be disabled by setting `on` to False.

    Usage:
        >>> @time_function
        >>> def func(x):
        >>>     pass
    """
    if not on:
        return f
    import time
    if name is None:
        name = f.__qualname__

    def f_timed(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print('{}: {} sec'.format(name, time.time() - start))
        return result
    return f_timed


class TimeoutError(Exception):
    """
    Raised when a function takes longer than the allowed time.
    """

    def __init__(self, value="Timed Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def timeout(seconds, *args):
    """
    Simple decorator to stop a function after a specified time limit.
    Adapted from
    https://stackoverflow.com/questions/35490555/python-timeout-decorator

    Example:
        >>> @timeout(10)
            def infloop():
                while true:
                    pass
            infloop()
            print('Exited the infinite loop!')
    """
    import signal
    import time

    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, handler)
            old_time_left = signal.alarm(seconds)
            if 0 < old_time_left < seconds:  # never lengthen existing timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
            finally:
                if old_time_left > 0:  # deduct f's run time from saved timer
                    old_time_left -= time.time() - start_time
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(old_time_left if old_time_left > 0 else 0)
            return result
        new_f.__name__ = f.__name__
        return new_f
    return decorate


def retry_enumerate(iterable, start=0, max_time=3600):
    """
    Wrapper around enumerate that retries if memory is unavailable.
    """
    import time
    retries = 0
    seconds = 0
    while True:
        try:
            return enumerate(iterable, start=start)
        except OSError:
            seconds = 2 ** retries
            if seconds >= max_time:
                raise
            print('Low on memory. Retrying in {} sec.'.format(seconds))
            time.sleep(seconds)
            retries += 1
            continue


def dilate_mask(mask, radius=5):
  return skmaximum(np.squeeze(mask).astype(np.uint8), skdisk(radius)).reshape(mask.shape).astype(np.bool)


def is_blank(image):
  """Check if image is blank (assumes ndarray or torch tensor only)
  """
  if isinstance(image, np.ndarray):
    return np.min(image) == 0 and np.max(image) == 0
  else:
    return torch.min(image) == 0 and torch.max(image) == 0


def invert(U, lr=0.1, max_iter=1000, currn=5, avgn=20, eps=1e-9):
  """Compute the inverse vector field of residual field U by optimization

  This method uses the following loss function:
  ```
  L = \frac{1}{2} \| U(V) - I \|^2 + \frac{1}{2} \| V(U) - I \|^2
  ```

  Args
     U: 4D tensor in vector field convention (1xXxYx2), where vectors are stored
        as absolute residuals.

  Returns
     V: 4D tensor for absolute residual vector field such that V(U) = I.
  """
  V = -deepcopy(U)
  if tensor_approx_eq(U,V):
    return V
  V.requires_grad = True
  n = U.shape[1] * U.shape[2]
  opt = torch.optim.SGD([V], lr=lr)
  costs = []
  currt = 0
  print('Optimizing inverse field')
  for t in range(max_iter):
    currt = t
    f = compose_fields(U, V)
    g = compose_fields(V, U)
    L = 0.5*torch.mean(f**2) + 0.5*torch.mean(g**2)
    costs.append(L)
    L.backward()
    V.grad *= n
    opt.step()
    opt.zero_grad()
    assert(not torch.isnan(costs[-1]))
    if costs[-1] == 0:
      break
    if len(costs) > avgn + currn:
        hist = sum(costs[-(avgn+currn):-currn]).item() / avgn
        curr = sum(costs[-currn:]).item() / currn
        if abs((hist-curr)/hist) < eps:
            break
  V.requires_grad = False
  print('Final cost @ t={0}: {1}'.format(currt, costs[-1].item()))
  return V

def get_chunk_dim(scale_factor):
  return scale_factor, scale_factor

def center_image(X, scale_factor, device=torch.device('cpu')):
  chunk_dim = get_chunk_dim(scale_factor)
  avg_pool = AvgPool2d(chunk_dim, stride=chunk_dim).to(device=device)
  X_bar_down = avg_pool(X)
  # X_bar = interpolate(X_bar_down, scale_factor=scale_factor, mode='nearest')
  X_bar = interpolate(X_bar_down, size=X.shape[2:], mode='nearest')
  return X - X_bar

def cpc(S, T, scale_factor, norm=True, device=torch.device('cpu')):
  chunk_dim = get_chunk_dim(scale_factor)
  sum_pool = LPPool2d(1, chunk_dim, stride=chunk_dim).to(device=device)
  S_hat = center_image(S, scale_factor, device=device)
  T_hat = center_image(T, scale_factor, device=device)
  if norm:
    S_hat_std = pow(sum_pool(pow(S_hat, 2)), 0.5)
    T_hat_std = pow(sum_pool(pow(T_hat, 2)), 0.5)
    norm = reciprocal(mul(S_hat_std, T_hat_std))
    R = mul(sum_pool(mul(S_hat, T_hat)), norm)
  else:
    R = sum_pool(mul(S_hat, T_hat))
  return R

def compute_distance(U, V):
  """Compute Euclidean distance between two field tensors

  Args:
    U: torch tensor for field
    V: torch tensor for field

  Returns:
    torch tensor (image format) with distance between U, V
  """
  D = U - V
  N = pow(D, 2)
  return pow(torch.sum(N, 3), 0.5).unsqueeze(0)

def vector_vote(fields, softmin_temp, blur_sigma=None):
  """Produce a single, consensus vector field from a set of vector fields

  Args:
    fields: list of fields (torch tensors in gridsample convention)
    softmin_temp: float for temperature of softmin
    blur_sigma: std dev of the Gaussian kernel by which to blur the inputs;
      default None means no blurring

  Returns:
    single vector field
  """
  if len(fields) == 1:
      return fields[0]

  print('softmin_temp {}'.format(softmin_temp))
  fields_blurred = fields
  if blur_sigma:
    fields_blurred = [blur_field(f, blur_sigma) for f in fields]
  n = len(fields)
  assert(n % 2 == 1)
  # majority
  m = n // 2 + 1
  indices = range(n)

  # compute distances for all pairs of fields
  dists = {}
  for i,j in combinations(indices, 2):
    dists[(i,j)] = compute_distance(fields_blurred[i], fields_blurred[j])
  # compute mean distance for all m-tuples
  mtuple_avg = []
  mtuples = list(combinations(indices, m))
  for mt in mtuples:
    t = []
    for i,j in combinations(mt, 2):
      t.append(dists[(i,j)])
    mtuple_avg.append(torch.mean(torch.cat(t, dim=0), dim=0, keepdim=True))
  mavg = torch.cat(mtuple_avg, dim=0)

  # compute weights for mtuples, giving higher weight to mtuples w/ smaller distances
  mtuple_weights = softmax(-mavg/softmin_temp , dim=0)
  # assign mtuple weights back to individual fields
  field_weights = torch.zeros((n,) + mtuple_weights.shape[1:])
  field_weights = field_weights.to(device=mtuple_weights.device)
  for i, mtuple in enumerate(mtuples):
    for j in mtuple:
      field_weights[j,...] += mtuple_weights[i,...]
  # divy up the weight by contribution
  field_weights = field_weights / m
  # create a voted field by multiplying fields by field weights
  field = torch.cat(fields, dim=0)
  field = field.permute(1,2,3,0)
  field_weights = field_weights.permute(2,3,0,1)
  return matmul(field, field_weights).permute(3,0,1,2)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Copied from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cuda'):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.weight = self.weight.to(device)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def blur_field(field, sigma, pad=(2,2,2,2)):
  smoothing = GaussianSmoothing(2, 5, 1, device=field.device)
  field = field.permute(0,3,1,2)
  field = F.pad(field, pad, mode='reflect')
  return smoothing(field).permute(0,2,3,1)
