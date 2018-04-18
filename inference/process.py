import torch
import numpy as np
from model.PyramidTransformer import PyramidTransformer
from model.xmas import Xmas

class Process(object):
    """docstring for Process."""
    def __init__(self, path='model_repository/9_4_2.pt', cuda=False, is_Xmas=False):
        super(Process, self).__init__()
        self.cuda = cuda
        self.is_Xmas = is_Xmas
        if self.is_Xmas:
            self.height = 4
            self.model = Xmas().load(archive_path=path, height=self.height, skips=0, cuda=cuda)
            self.convs = self.model.G_level
            self.skip = 0
            self.mip = 2
        else:
            self.height = 5
            self.model = PyramidTransformer.load(archive_path=path, height=self.height, skips=2, cuda=cuda)
            self.skip = self.model.pyramid.skip
            self.convs = self.model.pyramid.mlist
            self.mip = 4 # hardcoded to be the mip that the model was trained at

    def process(self, s, t, level=0, crop=0):
        if level < self.mip + self.skip or level > self.mip + self.height - 1:
            return None
        level -= self.mip
        #print("~ Level: ", level) #FU davit
        x = (t, s) if self.is_Xmas else (s, t)
        x = torch.from_numpy(np.stack(x, axis=1))
        if self.cuda:
            x = x.cuda()
        x = torch.autograd.Variable(x, requires_grad=False)
        res = self.convs[level](x)
        if self.is_Xmas:
            res = res.permute(0,2,3,1)
        if crop>0:
            res = res[:,crop:-crop, crop:-crop,:]
        if not self.is_Xmas:
            res *= 44 * (2 ** (self.mip + self.height - 1))
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
