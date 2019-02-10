import torch
from alignermodule import Aligner
from rollback_pyramid import RollbackPyramid

aligners = {}
pyramid = RollbackPyramid()

for m in [8, 9, 10]:
    aligners[m] = Aligner(fms=[2, 16, 16, 16, 16, 2], k=7).cuda()
    aligners[m].load_state_dict(torch.load('./checkpoints/barak_aligner_mip{}.pth.tar'.format(m)))
    pyramid.set_mip_processor(aligners[m], m)
