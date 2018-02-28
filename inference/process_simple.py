import torch
import numpy as np

class Process(object):
    """docstring for Process."""
    def __init__(self, path="model/1_6_3.pt", cuda=True):
        super(Process, self).__init__()
        self.model = torch.load(path)
        if cuda:
            self.model = self.model.cuda()

    def process(self, s, t, level=0):
        s = torch.from_numpy(s)
        t = torch.from_numpy(t)
        stack = torch.cat((s,t),1)
        residual = self.model().mlist[level](stack)
        return resiudal.data.cpu().numpy()

#simple test
if __name__ == "__main__":
    a = Process()
    s = numpy.ones((8,256,256, 1))
    t = numpy.ones((8,256,256, 1))
    flow = a.process(s, t)
    print(flow.shape)
