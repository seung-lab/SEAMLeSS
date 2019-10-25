import torch

class DataParallelAccessAttributes(torch.nn.DataParallel):
    """Subclass to access attributes of the wrapped Module
    """
    def __getattr__(self, name):
        return getattr(self.module, name)
