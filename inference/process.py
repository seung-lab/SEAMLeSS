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
        
    def contrast(self, t):
        zeromask = (t == 0)
        l,h = 145.0, 210.0
        t[t < l/255.0] = l/255.0
        t[t > h/255.0] = h/255.0
        t *= 255.0 / (h-l+1)
        t -= np.min(t)
        t += 1.0/255.0
        t[zeromask] = 0

    def process(self, s, t, level=0, crop=0):        
        if level != self.mip:
            return None
        if self.should_contrast:
            s = self.normalizer.apply(s.squeeze()).reshape(t.shape)
            t = self.normalizer.apply(t.squeeze()).reshape(t.shape)
            #self.contrast_(s)
            #self.contrast_(t)
        else:
            print('Skipping contrast...')
        level -= self.mip

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
        x = torch.autograd.Variable(x, requires_grad=False)
        image, field, residuals, encodings, cumulative_residuals = self.model(x)
        res = self.model(x)[1] - self.model.pyramid.get_identity_grid(x.size(3))
        res *= (res.shape[-2] - 1) / 2 * (2 ** self.mip)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        nonflipped = res.data.cpu().numpy()

        if not self.flip_average:
            return nonflipped, residuals, encodings, cumulative_residuals

        # flipped
        s = np.flip(s,1)
        s = np.flip(s,2)
        t = np.flip(t,1)
        t = np.flip(t,2)
        x = torch.from_numpy(np.stack((s,t), axis=1))
        if self.cuda:
            x = x.cuda()
        x = torch.autograd.Variable(x, requires_grad=False)
        image_fl, field_fl, residuals_fl, encodings_fl, cumulative_residuals_fl = self.model(x)
        res = self.model(x)[1] - self.model.pyramid.get_identity_grid(x.size(3))
        res *= (res.shape[-2] - 1) / 2 * (2 ** self.mip)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        res = res.data.cpu().numpy()
        res = np.flip(res,1)
        res = np.flip(res,2)
        flipped = -res
        
        return (flipped + nonflipped)/2.0, residuals, encodings, cumulative_residuals # TODO: include flipped resid & enc
#        return flipped, residuals_fl, encodings_fl, cumulative_residuals_fl # TODO: include flipped resid & enc

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
