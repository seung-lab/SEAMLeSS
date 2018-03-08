import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def mse_loss(inp, target):
    out = torch.pow((inp-target), 2)
    return torch.mean(out)

s = nn.Sigmoid()
def sigmoid_loss(inp, target):
    out = torch.pow((inp-target), 2)
    out = 1-s(-10*(out-0.01))
    return torch.mean(out)

def norm(x):
    return torch.sum(x)#torch.pow(x, 2))

def normalize(x):
    n = torch.pow(x, 2)
    n = torch.sum(n, 1)
    n = torch.sqrt(n)
    return x/n

def smoothness_penalty(fields, order=1):
    dx = lambda f: f[:,:,1:,:] - f[:,:,:-1,:]
    dy = lambda f: f[:,:,:,1:] - f[:,:,:,:-1]

    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given k-th derivatives, compute (k+1)-th
    square_errors = map(lambda f: torch.sum(f ** 2, -1), fields) # sum along last axis (x/y channel)
    return sum(map(torch.mean, square_errors))

import cavelab as cl
def loss(xs, ys, Rs, rs, label=0, start=0, lambda_1=0, lambda_2=0):
    crop = 30
    #r = normalize(Rs[-1])
    r = [[Rs[i][:,:,crop:-crop,crop:-crop]] for i in range(Rs.shape[0])] #[:, :,crop:-crop,crop:-crop]

    p1 = smoothness_penalty(r, 1)
    p2 = smoothness_penalty(r, 2)
    #cl.visual.save(ys[0,0,crop:-crop,crop:-crop].data.cpu().numpy(), 'dump/pred_loss')
    #cl.visual.save(xs[0,0,crop:-crop,crop:-crop].data.cpu().numpy(), 'dump/target_loss')

    mse = mse_loss(ys[:, :,crop:-crop,crop:-crop],
                   xs[:, :-1,crop:-crop,crop:-crop])
    loss = mse+lambda_1*p1+lambda_2*p2
    return loss, mse, lambda_1*p1, lambda_2*p2
