import torch
import numpy as np
from PyramidTransformer import PyramidTransformer

class Process(object):
    """docstring for Process."""
    def __init__(self, path='model/1_6_3.pt', cuda=False):
        super(Process, self).__init__()
        self.cuda = cuda
        self.model = PyramidTransformer().load(archive_path=path, cuda=cuda)

    def process(self, s, t, level=0):
        x = torch.from_numpy(np.stack((s, t), axis=1))
        if self.cuda:
            x = x.cuda()
        x = torch.autograd.Variable(x, requires_grad=False)
        res = self.model.pyramid.mlist[level](x)
        return res.data.cpu().numpy()

#Simple test
if __name__ == "__main__":
    a = Process()
    s = np.ones((8,256,256), dtype=np.float32)
    t = np.ones((8,256,256), dtype=np.float32)
    flow = a.process(s, t)
    print(flow.shape) #expected output (8,256,256,2)
