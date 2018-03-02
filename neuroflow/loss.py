import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def mse_loss(input,target, crop=1):
    out = torch.pow((input[:,crop:-crop,crop:-crop]-target[:,crop:-crop,crop:-crop]), 2)
    return out.mean()

downsample = nn.AvgPool2d(2, stride=2)

def smoothness_penalty(fields, label, order=1, mask=True):
    dx =     lambda f: (f[:,:,1:,:] - f[:,:,:-1,:])
    dy =     lambda f: (f[:,:,:,1:] - f[:,:,:,:-1])

    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given k-th derivatives, compute (k+1)-th
    fields = list(map(lambda f: torch.sum(f ** 2, 1), fields)) # sum along last axis (x/y channel)

    if mask:
        fields = [torch.mul(f, label[:,:f.shape[1],:f.shape[2]]) for f in fields]
    fields = [torch.mean(f) for f in fields]
    penalty = sum(fields)/len(fields)
    return penalty

def loss(xs, ys, Rs, rs, label, start=0, lambda_1=0, lambda_2=0):
    shp = xs[-1].shape
    level = len(xs)
    r_crop = 1 #2**level
    mse_crop = 1 #2**level

    r = F.upsample(rs[0], scale_factor=2**level, mode='nearest')
    for i in range(1, len(rs)):
        r = r + F.upsample(rs[i], scale_factor=2**(level-i), mode='nearest')
    res = r[:,:,r_crop:-r_crop,r_crop:-r_crop]

    label = Variable(torch.zeros_like(label.data).cuda(device=0), requires_grad=False)
    p1 = lambda_1*smoothness_penalty([res], 1-label, 1, mask=False)
    p2 = lambda_2*smoothness_penalty([res], 1-label, 2, mask=False)

    start = 0
    mse = 0
    for i in range(start, len(xs)):
        mse = mse_loss(ys[i][:,0,:,:],
                       xs[i][:,1,:,:],
                       crop=mse_crop)
    mse = mse/(len(xs)-start)
    loss = mse+p2+p1

    return loss, mse, p1, p2
