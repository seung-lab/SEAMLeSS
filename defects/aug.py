import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import time
import itertools

from helpers import reverse_dim

def half(a,b):
    return a if random.randint(0,1) == 0 else b

def apply_grid(stack, grid):
    for sliceidx in range(stack.size(1)):
        stack[:,sliceidx:sliceidx+1] = F.grid_sample(stack[:,sliceidx:sliceidx+1], grid)
    return stack

def rotate_and_scale(imslice, size=0.004, scale=0.005, grid=None):
    if type(imslice) == list:
        for _ in range(4 - len(imslice[0].size())):
            imslice = [o.unsqueeze(0) for o in imslice]
    else:
        for _ in range(4 - len(imslice.size())):
            imslice = imslice.unsqueeze(0)
    if grid is None:
        if size is None:
            theta = np.random.uniform(0, 2 * np.pi)
        else:
            theta = np.random.normal(0, size)
        scale = np.random.normal(1, scale)
        mat = torch.FloatTensor([[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0]]]) * scale
        grid = F.affine_grid(mat, imslice.size() if type(imslice) != list else imslice[0].size())
        if (imslice.is_cuda if type(imslice) != list else imslice[0].is_cuda):
            grid = grid.cuda()
    if type(imslice) == list:
        output = [apply_grid(o.clone(), grid) for o in imslice]
    else:
        output = apply_grid(imslice.clone(), grid)
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

    color_mean = np.random.uniform()
    outslice = imslice.clone()
    mask = Variable(torch.ones(outslice.size())).cuda()
    for idx, p in enumerate(pos):
        outslice.data[idx,:p-width] = outslice.data[idx,width:p]
        color = torch.cuda.FloatTensor(np.random.normal(color_mean, 0.2, width)).clamp(min=0,max=1)
        if torch.max(outslice.data[idx,width:p]) > 0:
            outslice.data[idx,p-width:p] = color
        mask.data[idx,p-width:p] = 0
    return outslice, mask

def jitter_stacks(Xs, max_displacement=2**6, min_cut=32):
    assert len(Xs) > 0
    should_rotate = random.randint(0,1) == 0
    srcXs = rotate_and_scale(Xs, size=None)[0] if should_rotate else Xs
    Xs_ = [Variable(torch.zeros(X.size())) for X in Xs]
    d = int(max_displacement / (2. * np.sqrt(2)))
    if Xs[0].is_cuda:
        Xs_ = [X_.cuda() for X_ in Xs_]
    for i in range(len(Xs)):
        Xs_[i][:,-1] = Xs[i][:,-1]
    for i in range(Xs[0].size()[1] - 1, -1, -1):
        xoff = half(weighted_draw(-d,-1), weighted_draw(1,d))
        yoff = half(weighted_draw(-d,-1), weighted_draw(1,d))
        for ii in range(len(Xs_)):
            if xoff >= 0:
                if yoff >= 0:
                    Xs_[ii][:,i,xoff:,yoff:] = srcXs[ii][:,i,:-xoff,:-yoff]
                else:
                    Xs_[ii][:,i,xoff:,:yoff] = srcXs[ii][:,i,:-xoff,-yoff:]
            else:
                if yoff >= 0:
                    Xs_[ii][:,i,:xoff,yoff:] = srcXs[ii][:,i,-xoff:,:-yoff]
                else:
                    Xs_[ii][:,i,:xoff,:yoff] = srcXs[ii][:,i,-xoff:,-yoff:]

            Xs_[ii][:,i:i+1] = rotate_and_scale(Xs_[ii][:,i:i+1])[0]
            cut_range = (min_cut, Xs_[ii].size(-1)//5)
            if ii == 0: # we only want to cut our images; we're assuming the images are first in Xs_, then masks 
                cut = random.randint(cut_range[0], cut_range[1])
                if random.randint(0,1) == 0:
                    Xs_[ii][:,i,:cut,:] = 0
                else:
                    Xs_[ii][:,i,-cut:,:] = 0

                cut = random.randint(cut_range[0], cut_range[1])
                if random.randint(0,1) == 0:
                    Xs_[ii][:,i,:,:cut] = 0
                else:
                    Xs_[ii][:,i,:,-cut:] = 0

    Xs_ = rotate_and_scale(Xs_, size=None)[0] if should_rotate else Xs_
    return Xs_

def gen_gradient(size, flip=None, period_median=25, peak=0.5):
    if flip is None:
        flip = half(True,False)
    periods = int(1 + np.random.exponential(-np.log(.5)*period_median))
    grad = torch.zeros(size)
    peak *= np.random.uniform(0,1)
    for period in range(periods):
        for idx in range(size[-1] // periods):
            if period % 2 == 0:
                grad[:, :, idx + period * (size[-1] // periods),:] = (float(idx) / (size[-1] / periods)) * peak
            else:
                grad[:, :, idx + period * (size[-1] // periods),:] = peak - (float(idx) / (size[-1] / periods)) * peak
    grad -= torch.mean(grad)
    grad = Variable(grad)
    if flip:
        grad = grad.permute(0,1,3,2)
    if random.randint(0,1) == 0:
        grad = reverse_dim(grad, 2)
    if random.randint(0,1) == 0:
        grad = reverse_dim(grad, 3)
    if random.randint(0,1) == 0:
        grad = -grad
    grad = rotate_and_scale(grad, None, 0.1)[0].squeeze()
    return grad.cuda()

def aug_brightness(X, factor=2.0, mask=False):
    # Assuming we get an input within [0,1]
    mi, ma = torch.min(X).data[0], torch.max(X).data[0]
    assert mi >= -1e-6 and ma <= 1 + 1e-6, 'Data must fall in range [0,1] ({}, {})'.format(mi, ma)
    
    gradients = random.randint(0,2)
    for _ in range(gradients):
        X = X + gen_gradient(X.size())

    X = X.clamp(min=0,max=1)
    compress = random.randint(0,1) == 0
    severity = np.random.uniform(1,factor)

    X = X 
    if compress:
        X = X / severity
    else:
        X = X * severity

    return X

"""
# Historical note:
# I don't understand why this doesn't work- an issue with PyTorch in-place operations?
def translate(chunk, x, y):
    chunk[:,:,x:] = chunk[:,:,:-x]
    chunk[:,:,:x] = 0
    chunk[:,:,:,y:] = chunk[:,:,:,:-y]
    chunk[:,:,:,:y] = 0

    return chunk
"""

def translate(chunk, x, y):
    out = torch.zeros(chunk.size())
    if chunk.is_cuda:
        out = out.cuda()
    out[:,:,:-x,:-y] = chunk[:,:,x:,y:]
    return out

def displace_slice(stack, slice_idx, aux, displacement=32):
    stack, grid = rotate_and_scale(stack, None)
    aux = [rotate_and_scale(a, grid=grid)[0] for a in aux]
    x = random.randint(1, displacement)
    y = random.randint(1, displacement)
    stack[:,slice_idx:slice_idx+1] = Variable(translate(stack.data[:,slice_idx:slice_idx+1], x, y))
    aux = [Variable(translate(a.data, x, y)) for a in aux]
    stack, grid = rotate_and_scale(stack, None)
    aux = [rotate_and_scale(a, grid=grid)[0].squeeze() for a in aux]
    return stack, aux

def weighted_draw(l,h,exp=2, wf=None):
    """
    Draw a value from the range [l,h] (inclusive) weighted such that
    the probability of drawing a value k is roughly proportional to the value
    itself (with an exponent to make the weighting less severe; exponent=0 gives
    a uniform distribution).
    """
    weight_function = lambda x: (abs(x) - min(abs(l), abs(h)) + 1) ** float(exp) if wf is None else wf
    vals = range(l,h+1)
    weights = np.array([weight_function(v) for v in vals])
    weights /= np.sum(weights)
    return np.random.choice(vals, p=weights)

def random_rect_mask(size):
    dimx = random.randint(1,size[-2]/2)
    dimy = random.randint(1,size[-2]/2)
    mx = size[-2]/2-dimx/2
    my = size[-2]/2-dimy/2
    prerotated_centered = Variable(torch.zeros(size)).cuda()
    prerotated_centered[mx:mx+dimx,my:my+dimy] = 1
    rotated_centered, _ = rotate_and_scale(prerotated_centered, None, 0)
    upper_bound = int(size[-2]/2 - max(dimx,dimy) / np.sqrt(2))
    dx = weighted_draw(1,upper_bound,1)
    dy = weighted_draw(1,upper_bound,1)
    off_centered = Variable(translate(rotated_centered.data, dx, dy))
    output = torch.ceil(rotate_and_scale(off_centered, None, 0)[0])
    return output.byte()

def aug_input(x, factor=2.0):
    zm = x == 0
    idx = random.randint(0,x.size()[0]-1)
    out = x if len(x.size()) == 2 else x[idx].clone()
    contrast_cutouts = half(0, random.randint(1,4))
    missing_cutouts = half(0, half(0,1))
    for _ in range(contrast_cutouts):
        mask = random_rect_mask(x.size())
        out[mask] = out[mask] / np.random.uniform(1,factor)
        out[mask] = out[mask] + np.random.uniform(0,1-torch.max(out[mask]).data[0])

    missing_masks = []
            
    for _ in range(missing_cutouts):
        mask = random_rect_mask(x.size())
        missing_masks.append(mask)
        out[mask] = np.random.uniform(0,1)

    if len(x.size()) == 3:
        out2 = x.clone()
        out2[idx] = out
        out = out2

    out = aug_brightness(out, factor)

    out[zm] = 0
    
    return out, missing_masks

def pad_stacks(stacks, total_padding):
    dim = stacks[0].size()[-1]
    top, left = random.randint(total_padding//4, total_padding-total_padding//4), random.randint(total_padding//4, total_padding-total_padding//4)
    padder = nn.ConstantPad2d((left, total_padding - left, top, total_padding - top), 0)
    return [padder(stack) for stack in stacks], top, left

def flip_stacks(Xs):
    if random.randint(0,1) == 0:
        Xs = [reverse_dim(X, 1) for X in Xs]
    if random.randint(0,1) == 0:
        Xs = [reverse_dim(X, 2) for X in Xs]
    if random.randint(0,1) == 0:
        Xs = [reverse_dim(X, 3) for X in Xs]
    if random.randint(0,1) == 0:
        Xs = [X.permute(0,1,3,2) for X in Xs]
    return Xs

def aug_stacks(Xs, jitter=True, pad=True, flip=True, padding=128, jitter_displacement=64):
    if jitter:
        Xs = jitter_stacks(Xs, max_displacement=jitter_displacement)
    top, left = 0, 0
    if pad:
        Xs, top, left = pad_stacks(Xs, padding)
    if flip:
        Xs = flip_stacks(Xs)
    return Xs, top, left
