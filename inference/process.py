import torch
import numpy as np
from model.PyramidTransformer import PyramidTransformer
from utilities.archive import ModelArchive
from model.xmas import Xmas
from normalizer import Normalizer
from utilities.helpers import save_chunk

class Process(object):
    """docstring for Process."""
    def __init__(self, archive, mip, dim=1280, size=7):
        super(Process, self).__init__()
        self.height = size
        self.archive = archive
        self.model = self.archive.model
        self.mip = mip
        self.dim = dim
        self.flip_average = False 

    @torch.no_grad()
    def process(self, s, t, level=0):
        """Run source & target image through SEAMLeSS net. Provide final
        vector field and intermediaries.

        Args:
           s: source tensor
           t: target tensor
           level: MIP of source & target images

        If flip averaging is on, run the net twice.
        The second time, flip the image 180 degrees.
        Then average the resulting (unflipped) vector fields.
        This eliminates the effect of any gradual drift.
        """
        # nonflipped
        unflipped = self.model(s, t)
        unflipped *= (unflipped.shape[-2] / 2) * (2 ** self.mip)

        if not self.flip_average:
            return unflipped

        # flipped
        s = s.flip([2, 3])
        t = t.flip([2, 3])
        field_fl = self.model(s, t)
        field_fl *= (field_fl.shape[-2] / 2) * (2 ** self.mip)
        flipped = -field_fl.flip([1,2])
        
        return (flipped + unflipped)/2.0

#Simple test
if __name__ == "__main__":
    print('Testing...')
    a = Process()
    s = np.ones((2,256,256), dtype=np.float32)
    t = np.ones((2,256,256), dtype=np.float32)

    flow = a.process(s, t, level=7)
    assert flow.shape == (2,256,256,2)

    flow = a.process(s, t, level=8)
    assert flow.shape == (2,236,236,2)

    flow = a.process(s, t, level=11)
    assert flow == None

    flow = a.process(s, t, level=1)
    assert flow == None

    print ('All tests passed.')
