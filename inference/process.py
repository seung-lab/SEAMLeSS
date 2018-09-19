import torch
import numpy as np
from model.PyramidTransformer import PyramidTransformer
from model.xmas import Xmas
from normalizer import Normalizer
from helpers import save_chunk

class Process(object):
    """docstring for Process."""
    def __init__(self, path, mip, cuda=True, is_Xmas=False, dim=1280, skip=0, topskip=0, size=7, contrast=True, flip_average=True, old_upsample=False):
        super(Process, self).__init__()
        self.cuda = cuda
        self.height = size
        self.model = PyramidTransformer.load(archive_path=path, height=self.height, skips=skip, topskips=topskip, cuda=cuda, dim=dim, old_upsample=old_upsample)
        self.skip = self.model.pyramid.skip
        self.convs = self.model.pyramid.mlist
        self.mip = mip
        self.dim = dim
        self.should_contrast = contrast
        self.normalizer = Normalizer(min(5, self.mip))
        self.flip_average = flip_average

    def process(self, s, t, level=0, crop=0):        
        if level != self.mip:
            return None
        if self.should_contrast:
            s = self.normalizer.apply(s.squeeze()).reshape(t.shape)
            t = self.normalizer.apply(t.squeeze()).reshape(t.shape)
        else:
            print('Skipping contrast...')

        '''
        Run the net twice.
        The second time, flip the image 180 degrees.
        Then average the resulting (unflipped) vector fields.
        This eliminates the effect of any gradual drift.
        '''
        # nonflipped
        x = torch.from_numpy(np.stack((s,t), axis=1))
        if self.cuda:
            x = x.cuda()
        res = self.model(x)[1]
        res *= res.shape[-2] / 2 * (2 ** self.mip)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        nonflipped = res.cpu().numpy()

        if not self.flip_average:
            return nonflipped

        # flipped
        s = np.flip(s,1)
        s = np.flip(s,2)
        t = np.flip(t,1)
        t = np.flip(t,2)
        x = torch.from_numpy(np.stack((s,t), axis=1))
        if self.cuda:
            x = x.cuda()
        res = self.model(x)[1]
        res *= res.shape[-2] / 2 * (2 ** self.mip)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        res = res.cpu().numpy()
        res = np.flip(res,1)
        res = np.flip(res,2)
        flipped = -res
        return (flipped + nonflipped)/2.0
        
#Simple test
if __name__ == "__main__":
    print('Testing...')
    a = Process()
    s = np.ones((2,256,256), dtype=np.float32)
    t = np.ones((2,256,256), dtype=np.float32)

    flow = a.process(s, t, level=7)
    assert flow.shape == (2,256,256,2)

    flow = a.process(s, t, level=8, crop=10)
    assert flow.shape == (2,236,236,2)

    flow = a.process(s, t, level=11)
    assert flow == None

    flow = a.process(s, t, level=1, crop=10)
    assert flow == None

    print ('All tests passed.')
