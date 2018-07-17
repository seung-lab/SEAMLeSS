import torch
import numpy as np
from model.PyramidTransformer import PyramidTransformer
from model.xmas import Xmas
from normalizer import Normalizer
from helpers import save_chunk

class Process(object):
    """docstring for Process."""
    def __init__(self, path, mip, cuda=True, is_Xmas=False, dim=1280, skip=0, topskip=0, size=7, contrast=True):
        super(Process, self).__init__()
        self.cuda = cuda
        self.height = size
        self.model = PyramidTransformer.load(archive_path=path, height=self.height, skips=skip, topskips=topskip, cuda=cuda, dim=dim)
        self.skip = self.model.pyramid.skip
        self.convs = self.model.pyramid.mlist
        self.mip = mip
        self.dim = dim
        self.should_contrast = contrast
        self.normalizer = Normalizer(self.mip)
        
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
        level -= self.mip
        x = torch.from_numpy(np.stack((s,t), axis=1))
        if self.cuda:
            x = x.cuda()
        x = torch.autograd.Variable(x, requires_grad=False)
        res = self.model(x)[1] - self.model.pyramid.get_identity_grid(x.size(3))
        res *= (res.shape[-2] / 2) * (2 ** self.mip)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        return res.data.cpu().numpy()

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
