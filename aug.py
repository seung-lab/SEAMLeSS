import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

from helpers import reverse_dim
from helpers import save_chunk

def rotate_and_scale(imslice, size=0.003, scale=0.005, grid=None):
    if size is None:
        theta = np.random.uniform(0, 2 * np.pi)
    else:
        theta = np.random.normal(0, size)
    scale = np.random.normal(1, scale)
    if grid is None:
        mat = torch.FloatTensor([[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0]]]) * scale
        grid = F.affine_grid(mat, imslice.size())
        if imslice.is_cuda:
            grid = grid.cuda()
    output = Variable(torch.zeros(imslice.size()))
    if imslice.is_cuda:
        output = output.cuda()
    for sliceidx in range(output.size(1)):
        output[:,sliceidx:sliceidx+1] = F.grid_sample(imslice[:,sliceidx:sliceidx+1], grid)
    return output, grid

def crack(imslice, width_range=(4,32)):
    width = np.random.randint(width_range[0], width_range[1])
    pos = [random.randint(imslice.size()[-1]/4,imslice.size()[-1]-imslice.size()[-1]/4)]
    prob = random.randint(4,10)
    left = random.randint(0,1) == 0
    for _ in range(imslice.size()[-1]-1):
        r = random.randint(0,prob)
        if r == 0:
            if left:
                pos.append(pos[-1] + 1)
            else:
                pos.append(pos[-1] - 1)
        else:
            pos.append(pos[-1])
        if pos[-1] <= width or pos[-1] >= imslice.size()[-1]:
            pos.pop()
            break
    #black = random.randint(0,1) == 0
    color_mean = np.random.uniform()
    outslice = imslice.clone()
    mask = Variable(torch.ones(outslice.size())).cuda()
    for idx, p in enumerate(pos):
        outslice.data[idx,:p-width] = outslice.data[idx,width:p]
        color = torch.cuda.FloatTensor(np.random.normal(color_mean, 0.2, width)).clamp(min=0,max=1)#np.random.uniform(0.01, 0.15) if black else np.random.uniform(0.85, 0.99)
        if torch.max(outslice.data[idx,width:p]) > 0:
            outslice.data[idx,p-width:p] = color
        mask.data[idx,p-width:p] = 0
    return outslice, mask

def jitter_stack(X, displacement=32, cut_range=(32,72)):
    should_rotate = random.randint(0,1) == 0
    srcX = rotate_and_scale(X, size=None)[0] if should_rotate else X
    X_ = Variable(torch.zeros(X.size()))
    if X.is_cuda:
        X_ = X_.cuda()

    X_[:,-1] = X[:,-1]
    for i in range(X.size()[1] - 1, -1, -1):
        xoff = 0
        while xoff == 0:
            xoff = random.randint(-displacement, displacement)
        yoff = 0
        while yoff == 0:
            yoff = random.randint(-displacement, displacement)
        if xoff >= 0:
            if yoff >= 0:
                X_[:,i,xoff:,yoff:] = srcX[:,i,:-xoff,:-yoff]
            else:
                X_[:,i,xoff:,:yoff] = srcX[:,i,:-xoff,-yoff:]
        else:
            if yoff >= 0:
                X_[:,i,:xoff,yoff:] = srcX[:,i,-xoff:,:-yoff]
            else:
                X_[:,i,:xoff,:yoff] = srcX[:,i,-xoff:,-yoff:]

        X_[:,i:i+1] = rotate_and_scale(X_[:,i:i+1])[0]
        cut = random.randint(cut_range[0], cut_range[1])
        r = random.randint(4,7)
        if r == 4:
            X_[:,i,:cut,:] = 0
        elif r == 5:
            X_[:,i,-cut:,:] = 0
        elif r == 6:
            X_[:,i,:,:cut] = 0
        elif r == 7:
            X_[:,i,:,-cut:] = 0
        X_[:,i:i+1] = rotate_and_scale(X_[:,i:i+1])[0]

    return rotate_and_scale(X_, size=None)[0] if should_rotate else X_

def gen_gradient(size, flip=True, periods=1, peak=1):
    grad = torch.zeros(size)
    peak *= np.random.uniform(0,1)
    for period in range(periods):
        for idx in range(size[-1] // periods):
            if period % 2 == 0:
                grad[idx + period * (size[-1] // periods),:] = (float(idx) / (size[-1] / periods)) * peak
            else:
                grad[idx + period * (size[-1] // periods),:] = peak - (float(idx) / (size[-1] / periods)) * peak
    grad -= torch.mean(grad)
    grad = Variable(grad)
    if flip:
        grad = grad.permute(1,0)
    if random.randint(0,1) == 0:
        grad = reverse_dim(grad, 0)
    if random.randint(0,1) == 0:
        grad = reverse_dim(grad, 1)
    if random.randint(0,1) == 0:
        grad = -grad
    grad = rotate_and_scale(grad.unsqueeze(0).unsqueeze(0), None, 0.1)[0].squeeze()
    return grad.cuda()

def aug_brightness(X):
    zero_mask = X==0
    if random.randint(0,1) == 0:
        flip = random.randint(0,1) == 0
        periods = random.randint(1,10)
        X = X + gen_gradient(X.size(), flip, periods)

    r = random.randint(0,3)
    if r == 0:
        X = X / np.random.uniform(1,5)
    elif r == 1:
        X = 1 - (1 - X) / np.random.uniform(1,5)
    elif r == 2:
        X = X / np.random.uniform(0.2,1)
    elif r == 3:
        X = 1 - (1 - X) / np.random.uniform(0.2,1)
    if random.randint(0,1) == 0:
        flip = random.randint(0,1) == 0
        periods = random.randint(1,10)
        X = X + gen_gradient(X.size(), flip, periods)
    X[zero_mask] = 0
    return X
        
def aug_input(x):
    idx = random.randint(0,x.size()[0]-1)
    out = x if len(x.size()) == 2 else x[idx].clone()
    squares = random.randint(0,5)
    for _ in range(squares):
        dimx = random.randint(1,x.size()[-2]/4)
        dimy = random.randint(1,x.size()[-2]/4)
        rx = random.randint(0, x.size()[-2] - dimx)
        ry = random.randint(0, x.size()[-2] - dimy)
        r = random.randint(0,3)
        if r == 0:
            out[rx:rx+dimx,ry:ry+dimy] = out[rx:rx+dimx,ry:ry+dimy] / np.random.uniform(1,5)
        elif r == 1:
            out[rx:rx+dimx,ry:ry+dimy] = 1 - (1 - out[rx:rx+dimx,ry:ry+dimy]) / np.random.uniform(1,5)
        elif r == 2:
            out[rx:rx+dimx,ry:ry+dimy] = 0
        elif r == 3:
            out[rx:rx+dimx,ry:ry+dimy] = 1

    if len(x.size()) == 3:
        out2 = x.clone()
        out2[idx] = out
        out = out2

    out = aug_brightness(out)

    return out

def pad_stack(stack, total_padding):
    dim = stack.size()[-1]
    top, left = random.randint(total_padding//4, total_padding-total_padding//4), random.randint(total_padding//4, total_padding-total_padding//4)
    padder = nn.ConstantPad2d((left, total_padding - left, top, total_padding - top), 0)
    return padder(stack)

def flip_stack(X):
    if random.randint(0,1) == 0:
        X = reverse_dim(X, 1)
    if random.randint(0,1) == 0:
        X = reverse_dim(X, 2)
    if random.randint(0,1) == 0:
        X = reverse_dim(X, 3)
    if random.randint(0,1) == 0:
        X = X.permute(0,1,3,2)
    return X

def aug_stack(X, jitter=True, pad=True, flip=True, padding=128, jitter_displacement=32):
    if jitter:
        X = jitter_stack(X, displacement=jitter_displacement)
    if pad:
        X = pad_stack(X, padding)
    if flip:
        X = flip_stack(X)
    return X
