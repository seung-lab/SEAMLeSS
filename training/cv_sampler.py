import cloudvolume as cv
import numpy as np

class Sampler(object):
    def __init__(self, source='gs://neuroglancer/pinky40_v11/image', mip=5, dim=1152, height=10, zs=1, ze=1000, test_fraction=0.2):
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
        self.zs = zs
        self.ze = ze
        self.test_fraction = test_fraction

    def chunk_at_global_coords(self, xyz, xyz_):
        assert xyz[2] >= self.zs
        assert xyz_[2] <= self.ze
        factor = 2 ** self.vol.mip
        x, x_ = xyz[0]/factor, xyz_[0]/factor
        y, y_ = xyz[1]/factor, xyz_[1]/factor
        z, z_ = xyz[2], xyz_[2]
        squeezed = None
        try:
            squeezed = np.squeeze(self.vol[x:x_, y:y_, z:z_])
        except Exception as e:
            print('Exception {}'.format(e))
        if squeezed is not None:
            print('Chunk shape {}'.format(squeezed.shape))
        return squeezed

    def random_sample(self, train=True, offsets=None, size=None, split=True):
        if offsets is None:
            offsets = self.vol_offsets[:2]
        if size is None:
            size = self.vol_size[:2]

        frac = self.test_fraction if split else 0
        zs = self.zs if train else self.ze - (self.ze - self.zs) * frac
        ze = self.ze - (self.ze - self.zs) * frac - self.stack_height if train else self.ze - self.stack_height
        z = np.random.randint(zs,ze)
        x = np.random.randint(offsets[0], offsets[0] + size[0] - self.adj_dim)
        y = np.random.randint(offsets[1], offsets[1] + size[1] - self.adj_dim)

        this_chunk = self.chunk_at_global_coords((x,y,z), (x+self.adj_dim,y+self.adj_dim,z+self.stack_height))
        print(train, x, y, z, zs, ze)
        return this_chunk, (x,y,z,self.adj_dim,self.stack_height)
