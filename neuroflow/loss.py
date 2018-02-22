import torch
import torch.nn as nn

def mse_loss(input,target, crop=1):
    out = torch.pow((input[:,crop:-crop,crop:-crop]-target[:,crop:-crop,crop:-crop]), 2)
    return out.mean()


downsample = nn.AvgPool2d(2, stride=2)

def smoothness_penalty(fields, labels, order=1, mask=True):
    factor = lambda f: f.size()[2] / 256
    dx =     lambda f: (f[:,:,1:,:] - f[:,:,:-1,:])*factor(f)
    dy =     lambda f: (f[:,:,:,1:] - f[:,:,:,:-1])*factor(f)

    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given k-th derivatives, compute (k+1)-th
    fields = list(map(lambda f: torch.sum(f ** 2, 1), fields)) # sum along last axis (x/y channel)

    penalty = 0
    for i in range(len(labels)):
        for idx in range(2**order):
            f = fields[2**order*i+idx]
            if mask:
                f = torch.mul(f, labels[i][:,:f.shape[1],:f.shape[2]])
            penalty += torch.mean(f)

    return penalty/len(fields)

def loss(xs, ys, Rs, rs, label, start=0, lambda_1=0, lambda_2=0):

    labels = [1-label]
    for i in range(len(Rs)-1):
        labels.append(downsample(labels[-1]))
    labels.reverse()

    p1 = lambda_1*smoothness_penalty([Rs[-1]], [labels[-1]], 1, mask=False)
    p2 = lambda_2*smoothness_penalty([Rs[-1]], [labels[-1]], 2, mask=False)

    start = 0
    mse = 0
    for i in range(start, len(xs)):
        mse += mse_loss(ys[i][:,0,:,:],
                        xs[i][:,1,:,:],
                        crop=2**i)
    mse = mse/(len(xs)-start)
    loss = mse+p2+p1
    return loss, mse, p1, p2
