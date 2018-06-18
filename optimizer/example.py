from cloudvolume import CloudVolume as cv
import numpy as np
import torch
from torch.autograd import Variable
from optimize import Optimizer
from copy import deepcopy
from scipy.misc import imresize
#src = "gs://neuroglancer/pinky40_alignment/prealigned"
#src = "gs://neuroglancer/pinky40_alignment/prealigned"
src = "gs://neuroglancer/basil_v0/father_of_alignment/v3"
target = "gs://neuroglancer/nflow_tests/optimizer_basil_7_without_mask"
cracks = "gs://neuroglancer/basil_v0/father_of_alignment/v3/mask/crack_detector_v3"
folds = "gs://neuroglancer/basil_v0/father_of_alignment/v3/mask/fold_detector_v1"

mip = 2
mask_mip  = 5
size = 2048
dpth = 2
x, y, init_z = 98381, 237865, 886 #41938, 27922, 18
m_xyz = (x/(2**mask_mip), y/(2**mask_mip), init_z)
m_size = size/(2**(mask_mip-mip))
x, y, init_z = x/(2**mip), y/(2**mip), init_z #103186-size*(2**(mip-1)), 242074-size*(2**(mip-1)), 877


# Cloud Volume business
src_info = cv(src, parallel=True).info
dst_info = deepcopy(src_info)
cv(target, info=dst_info, parallel=True).commit_info()
src = cv(src, parallel=True, mip=mip)
cracks = cv(cracks, parallel=True, mip=mask_mip)
folds = cv(folds, parallel=True, mip=mask_mip)
trg = cv(target, parallel=False, mip=mip, non_aligned_writes=True,
                fill_missing=True,
                cdn_cache=False)

flow = Optimizer(ndownsamples=5, currn=5, avgn=20, lambda1=0.5, lr=0.1, eps=0.001, min_iter=100, max_iter=5000)

image = src[x:x+size,y:y+size,init_z]
trg[x:x+size,y:y+size,init_z] = image
target = image[:,:,0,0]/255.0
mask = np.zeros_like(target)

for i in range(10):
    z = init_z + i
    # Load the slice
    source = src[x:x+size,y:y+size,z+1]/255.0
    source = source[:,:,0,0]

    # Load the mask
    fold = folds[m_xyz[0]:m_xyz[0]+m_size, m_xyz[1]:m_xyz[1]+m_size, z+1]
    crack = cracks[m_xyz[0]:m_xyz[0]+m_size, m_xyz[1]:m_xyz[1]+m_size, z+1]
    mask = np.minimum(fold, crack)[:,:,0,0]
    mask = imresize(mask, size=(size, size))/255.0

    # Compute the flow and render
    field = flow.process(source, target, mask=mask)
    pred = flow.render(source, field)
    target = pred[0,0]

    # Upload
    trg[x:x+size,y:y+size,z+1] = (pred[0,0,:,:,np.newaxis, np.newaxis]*255).astype(np.uint8)
