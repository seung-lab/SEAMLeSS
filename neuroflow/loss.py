import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cavelab as cl

def mse_loss(inp, target, mask=1):

    out = torch.pow(torch.mul(inp-target, mask), 2)
    return torch.mean(out)


def normalize(x):
    norm = torch.sqrt(torch.sum(torch.pow(x,2)))
    return x/norm


def smoothness_penalty(fields, order=1):
    dx = lambda f: f[:,:,1:,:] - f[:,:,:-1,:]
    dy = lambda f: f[:,:,:,1:] - f[:,:,:,:-1]

    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    square_errors = map(lambda f: torch.sum(f ** 2, 1), fields) # sum along last axis (x/y channel)

    return sum(map(torch.mean, square_errors))


def loss(xs, ys, Rs, rs, level=0, lambda_1=0, lambda_2=0):
    crop = 16
    #r = normalize(Rs[-1])
    r = [Rs[level][:,:,crop:-crop,crop:-crop]]# for level in range(Rs.shape[0])] #[:, :,crop:-crop,crop:-crop]
    #print(ys[level, :,crop:-crop,crop:-crop].shape)
    #exit()
    p1 = smoothness_penalty(r, 1)
    #p2 = smoothness_penalty(r, 2)

    #cl.visual.save(ys[level, :,crop:-crop,crop:-crop].data.cpu().numpy(), 'dump/pred_loss')
    #cl.visual.save(xs[level, :-1,crop:-crop,crop:-crop].data.cpu().numpy(), 'dump/target_loss')
    #mask = xs[level,:,crop:-crop,crop:-crop]
    #mask = (mask[:-1]==0)==(mask[1:]==0)

    mse = mse_loss(ys[level, :,crop:-crop,crop:-crop],
                   xs[level, :-1,crop:-crop,crop:-crop])
                   #mask = mask.type(torch.cuda.FloatTensor))

    loss = mse+lambda_1*p1#+lambda_2*p2
    return loss, mse, p1, 0 #p2
