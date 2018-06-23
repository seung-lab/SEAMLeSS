import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from moviepy.editor import ImageSequenceClip
import collections
import torch
from torch.autograd import Variable

def get_colors(angles, f, c):
    colors = f(angles)
    colors = c(colors)
    return colors

def display_v(vfield, name):
    dim = vfield.shape[-2]
    assert type(vfield) == np.ndarray

    lengths = np.squeeze(np.sqrt(vfield[:,:,:,0] ** 2 + vfield[:,:,:,1] ** 2))
    lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
    angles = np.squeeze(np.angle(vfield[:,:,:,0] + vfield[:,:,:,1]*1j))

    scolors = get_colors(angles, f=np.sin, c=cm.magma)
    ccolors = get_colors(angles, f=np.cos, c=cm.Wistia)

    # mix
    scolors[:,:,0] = ccolors[:,:,0]
    scolors[:,:,2] = (ccolors[:,:,2] + scolors[:,:,2]) / 2
    scolors[:,:,-1] = lengths

    plt.imsave(name + '.png', scolors)

def reverse_dim(var, dim):
    idx = range(var.size()[dim] - 1, -1, -1)
    idx = Variable(torch.LongTensor(idx))
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

def save_chunk(chunk, name):
    plt.imsave(name + '.png', 1 - chunk, cmap='Greys')

def gif(filename, array, fps=8, scale=1.0):
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

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps, verbose=False)
    return clip
