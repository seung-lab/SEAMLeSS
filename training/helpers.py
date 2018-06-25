import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from moviepy.editor import ImageSequenceClip
import collections
import torch
from torch.autograd import Variable
from skimage.transform import rescale

def get_colors(angles, f, c):
    colors = f(angles)
    colors = c(colors)
    return colors

def dv(vfield, name=None):
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

    if name is not None:
        plt.imsave(name + '.png', scolors)
    else:
        return scolors

def np_upsample(img, factor):
    if img.ndim == 2:
        return rescale(img, factor)
    elif img.ndim == 3:
        b = np.empty((img.shape[0] * factor, img.shape[1] * factor, img.shape[2]))
        for idx in range(img.shape[2]):
            b[:,:,idx] = np_upsample(img[:,:,idx], factor)
        return b
    else:
        crash
                     
def display_v(vfield, name=None):
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
    
def dvl(V_pred, name):
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
