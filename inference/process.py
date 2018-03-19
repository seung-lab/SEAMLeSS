import torch
import numpy as np
from PyramidTransformer import PyramidTransformer
from XmasTransformer import *
class Process(object):
    """docstring for Process."""
    def __init__(self, path='model/2_5_2.pt', cuda=False):
        super(Process, self).__init__()
        self.cuda = cuda
        #print(Xmas())
        #self.model = torch.load(path)
        #print(self.model)
        #exit()
        self.model = PyramidTransformer().load(archive_path=path, height=5, skips=2, cuda=cuda)
        self.mip = 5 # hardcoded to be the mip that the model was trained at

    def process(self, s, t, level=0, crop=0):
        if level < self.mip + self.model.pyramid.skip:
            return None
        level -= self.mip
        x = torch.from_numpy(np.stack((s, t), axis=1))
        if self.cuda:
            x = x.cuda()
        x = torch.autograd.Variable(x, requires_grad=False)
        res = self.model.pyramid.mlist[level](x)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        return res.data.cpu().numpy()

#Simple test
if __name__ == "__main__":
    a = Process()
    s = np.ones((2,256,256), dtype=np.float32)
    t = np.ones((2,256,256), dtype=np.float32)

    flow = a.process(s, t, level=8)
    assert flow.shape == (2,256,256,2)

    flow = a.process(s, t, level=7, crop=10)
    assert flow.shape == (2,236,236,2)

    flow = a.process(s, t, level=6)
    assert flow == None

    flow = a.process(s, t, level=5, crop=10)
    assert flow == None

    print ('All tests passed.')
