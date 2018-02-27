import torch

class Process(object):
    """docstring for Process."""
    def __init__(self, path=""):
        super(Process, self).__init__()
        self.model = torch.load(path)
        #self.model.cuda()

    def process(self, s, t, mip=0):
        #FIXME Eric implement this functions
        # Feel free to modify the rest of the code
        # y, R, r = self.model(s,t)
        return y, R, r
