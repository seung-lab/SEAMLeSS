import numpy as np
#import cloudvolume as cv
import matplotlib.pyplot as plt
import os
import collections
import torch
from torch.autograd import Variable

GS_ALIGNED = 'gs://neuroglancer/pinky40_v11/image'
GS_PREALIGNED = 'gs://neuroglancer/pinky40_alignment/prealigned'

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

def load_volumes():
    return (cv.CloudVolume(GS_PREALIGNED), cv.CloudVolume(GS_ALIGNED))

def load_mip(mip, volume=GS_ALIGNED):
    return cv.CloudVolume(volume, mip=mip)

def snapshot_slice(volume, slice_range, name='slice'):
    """
    Takes a cloud volume and a numpy range such as
    np.s_[20000:22000, 22000:24000, 512] and saves it
    as a png.
    """
    plt.imsave('img/' + name + '.png', volume[slice_range][:,:,0,0])

def chunk_at_global_coords(volume, xyz, xyz_):
    factor = 2 ** volume.mip
    x, x_ = xyz[0]/factor, xyz_[0]/factor
    y, y_ = xyz[1]/factor, xyz_[1]/factor
    z, z_ = xyz[2], xyz_[2]
    #print 'translated:', xyz, xyz_, '->', (x,y,z), (x_, y_, z_)
    return np.swapaxes(np.squeeze(volume[x:x_, y:y_, z:z_]), 0, 1)

def chunks_in_mip_range(r, xyz=(30600,21500,118), xyz_=(52100,42300,119)):
    slices = []
    for mip in r:
        try:
            volume = load_mip(mip)
            slices.append(chunk_at_global_coords(volume, xyz, xyz_))
        except Exception as e:
            #print e
            pass
    return slices

def save_chunk(chunk, name):
    plt.imsave(name + '.png', 1 - chunk, cmap='Greys')

def save_chunks(chunks, prefix='img'):
    """Saves an array of 2d chunks in sequential images.
    Accepts arrays in the form (z, y, x).
    """
    for idx in range(chunks.shape[0]):
        save_chunk(chunks[idx,:,:], prefix + str(idx) + '.png')