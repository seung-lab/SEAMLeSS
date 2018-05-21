import cloudvolume as cv
import numpy as np
import torch
from torch.autograd import Variable

class Sampler(object):
    def __init__(self, source='gs://neuroglancer/pinky40_v11/image', mip=1, dim=512, height=128):
        import httplib2shim
        httplib2shim.patch()
        self.source = source
        self.vol = cv.CloudVolume(source, mip=mip)
        self.dim = dim
        self.vol_info = self.vol.info['scales'][0]
        self.vol_size = self.vol_info['size']
        self.vol_offsets = self.vol_info['voxel_offset']
        self.adj_dim = self.dim * 2 ** self.vol.mip
        self.stack_height = height
        
    def chunk_at_global_coords(self, xyz, xyz_):
        factor = 2 ** self.vol.mip
        x, x_ = xyz[0]/factor, xyz_[0]/factor
        y, y_ = xyz[1]/factor, xyz_[1]/factor
        z, z_ = xyz[2], xyz_[2]
        squeezed = None
        try:
            squeezed = np.squeeze(self.vol[x:x_, y:y_, z:z_])
        except Exception as e:
            print(e)
        if squeezed is not None:
            print(squeezed.shape)
        return squeezed

    def random_sample(self, train=True, offsets=None, size=None, split=True):
        if offsets is None:
            offsets = self.vol_offsets
        if size is None:
            size = self.vol_size
        print('params:', offsets, size)
        if 'basil' in self.source:
            offsets = (offsets[0] + 50000, offsets[1] + 50000, offsets[2])
            size = (size[0] / 2, size[1] / 2, size[2])
        print('adjusted params:', offsets, size)
        
        x = np.random.randint(offsets[0], offsets[0] + size[0] - self.adj_dim)
        y = np.random.randint(offsets[1], offsets[1] + size[1] - self.adj_dim)
        trainf = lambda x: x < 700
        testf = lambda x: x >= 800
        if train:
            z = np.random.randint(offsets[2], offsets[2] + size[2] - 1)
            while not trainf(z):
                z = np.random.randint(offsets[2], offsets[2] + size[2] - 1)
        else:
            z = np.random.randint(offsets[2], offsets[2] + size[2] - 1)
            while not testf(z):
                z = np.random.randint(offsets[2], offsets[2] + size[2] - 1)

        this_chunk = self.chunk_at_global_coords((x,y,z), (x+self.adj_dim,y+self.adj_dim,z+self.stack_height))
        print(train, x, y, z, self.adj_dim, self.stack_height)
        return this_chunk, (x,y,z,self.adj_dim,self.stack_height)
