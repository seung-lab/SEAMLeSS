import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def mse_loss(inp, target, crop=1):
    out = torch.pow((inp[:,crop:-crop,crop:-crop]-target[:,crop:-crop,crop:-crop]), 2)
    #out = torch.mul(out, 1-label[:,crop:-crop,crop:-crop])
    return out.mean()

def norm(x):
    return torch.sum(torch.pow(x, 2))

def normalize(x):
    n = torch.pow(x, 2)
    n = torch.sum(n, 1)
    n = torch.sqrt(n)
    return x/n

def smoothness_penalty(fields, order=1):
    dx = lambda f: (f[:,:,1:,:] - f[:,:,:-1,:])
    dy = lambda f: (f[:,:,:,1:] - f[:,:,:,:-1])

    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given k-th derivatives, compute (k+1)-th
    fields = list(map(lambda f: torch.sum(f ** 2, 1), fields)) # sum along last axis (x/y channel)

    fields = [norm(f) for f in fields]
    penalty = sum(fields)
    return penalty

def smoothness(rs, label, crop=1):
    r = rs[0]
    for i in range(1, len(rs)):
        r = rs[i]+ F.upsample(r, scale_factor=2, mode='nearest')

    #r = torch.mul(r, 1-label)
    res = r#[:,:,crop:-crop,crop:-crop]

    p1 = smoothness_penalty([res], 1)
    return p1

def loss(xs, ys, Rs, rs, label, start=0, lambda_1=0, lambda_2=0):
    crop = 2**(len(rs))*15
    #r = normalize(Rs[-1])
    r = Rs[-1]#[:, :,crop:-crop,crop:-crop]
    p1 = smoothness_penalty([r], 1)
    p2 = smoothness_penalty([r], 2)

    mse = mse_loss(ys[-1][:,0,:,:],
                   xs[-1][:,1,:,:],
                   crop=crop)
    loss = mse+lambda_1*p1+lambda_2*p2
    return loss, mse, lambda_1*p1, 0#lambda_2*p1
