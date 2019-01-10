import torch
import numpy as np
from model.PyramidTransformer import PyramidTransformer
from utilities.archive import ModelArchive
from model.xmas import Xmas
from normalizer import Normalizer
from utilities.helpers import save_chunk

class Process(object):
    """docstring for Process."""
    def __init__(self, archive, mip, cuda=True, is_Xmas=False, dim=1280, skip=0, topskip=0, size=7, contrast=True, flip_average=True, old_upsample=False):
        super(Process, self).__init__()
        self.cuda = cuda
        self.height = size
        self.archive = archive
        self.model = self.archive.model
        # self.model = PyramidTransformer.load(archive_path=path, height=self.height, skips=skip, topskips=topskip, cuda=cuda, dim=dim, old_upsample=old_upsample)
        self.mip = mip
        self.dim = dim
        self.should_contrast = contrast
        self.normalizer = self.archive.preprocessor
        # self.normalizer = Normalizer(min(5, self.mip))
        self.flip_average = flip_average

    @torch.no_grad()
    def process(self, s, t, level=0, crop=0):
        """
        Run a net on a pair of images and return the result.

        If flip averaging is on, run the net twice.
        The second time, flip the image 180 degrees.
        Then average the resulting (unflipped) vector fields.
        This eliminates the effect of any gradual drift.
        """
        if level != self.mip:
            return None
        s, t = torch.from_numpy(s).unsqueeze(0), torch.from_numpy(t).unsqueeze(0)
        if self.should_contrast and self.normalizer:
            s = self.normalizer(s).reshape(t.shape)
            t = self.normalizer(t).reshape(t.shape)
        else:
            print('Skipping contrast...')
        if self.cuda:
            s, t = s.cuda(), t.cuda()

        # nonflipped
        field, residuals, encodings, cumulative_residuals = self.model(s, t), *[None]*3
        field *= (field.shape[-2] / 2) * (2 ** self.mip)
        if crop>0:
            field = field[:,crop:-crop, crop:-crop,:]
        nonflipped = field.cpu().numpy()

        if not self.flip_average:
            return nonflipped, residuals, encodings, cumulative_residuals

        # flipped
        s = s.flip([2, 3])
        t = t.flip([2, 3])
        field_fl, residuals_fl, encodings_fl, cumulative_residuals_fl = self.model(s, t), *[None]*3
        field_fl *= (field_fl.shape[-2] / 2) * (2 ** self.mip)
        if crop>0:
            field_fl = field_fl[:,crop:-crop, crop:-crop,:]
        flipped = -field_fl.flip([1, 2])
        flipped = flipped.cpu().numpy()

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
