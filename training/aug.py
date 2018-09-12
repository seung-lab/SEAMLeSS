import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import time
import itertools

from helpers import reverse_dim, save_chunk, gif

def half(a=None,b=None):
    if a is None and b is None:
        a,b = True,False
    return a if random.randint(0,1) == 0 else b

def apply_grid(stack, grid):
    for sliceidx in range(stack.size(1)):
        stack[:,sliceidx:sliceidx+1] = F.grid_sample(stack[:,sliceidx:sliceidx+1], grid, mode='bilinear')
    return stack

def rotate_and_scale(imslice, size=0.01, scale=0.01, grid=None):
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
        # if (imslice.is_cuda if type(imslice) != list else imslice[0].is_cuda):
        #     grid = grid.cuda()
    if type(imslice) == list:
        output = [apply_grid(o.clone(), grid) for o in imslice]
    else:
        output = apply_grid(imslice.clone(), grid)
    return output, grid

def crack(imslice, width_range=(4,32)):
    width = np.random.randint(width_range[0], width_range[1])
    pos = [random.randint(imslice.size()[-1]//4,imslice.size()[-1]-imslice.size()[-1]//4)]
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

def random_translation(src, max_displacement=2**6):
    """Shift src by x & y up to max_displacement, keeping src size

    Args:
    * img: 2D array
    """
    dst = torch.zeros(src.size())
    d = int(max_displacement / (2. * np.sqrt(2)))
    xoff = weighted_draw(1,d) * half(1,-1)
    yoff = weighted_draw(1,d) * half(1,-1)
    if xoff >= 0:
        if yoff >= 0:
            dst[xoff:,yoff:] = src[:-xoff,:-yoff]
        else:
            dst[xoff:,:yoff] = src[:-xoff,-yoff:]
    else:
        if yoff >= 0:
            dst[:xoff,yoff:] = src[-xoff:,:-yoff]
        else:
            dst[:xoff,:yoff] = src[-xoff:,-yoff:]
    return dst

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

    xoff, yoff = None, None
    for i in range(Xs[0].size()[1] - 1, -1, -1):
        jitter = half()
        if jitter or xoff is None or yoff is None:
            xoff = weighted_draw(1,d) * half(1,-1)
            yoff = weighted_draw(1,d) * half(1,-1)
        for ii in range(len(Xs_)):
            if jitter:
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
            else:
                Xs_[ii][:,i] = srcXs[ii][:,i]

            cut_range = (min_cut, Xs_[ii].size(-1)//3)
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

def gen_gradient(size, flip=None, period_median=25, peak=0.3, randomize_peak=True, rotate=True):
    if flip is None:
        flip = half(True,False)
    if period_median is not None:
        periods = int(1 + np.random.exponential(-np.log(.5)*period_median))
    else:
        periods = 1
    grad = torch.zeros(size)
    if randomize_peak:
        peak *= np.random.uniform(0,1)
    for period in range(periods):
        for idx in range(size[-1] // periods):
            if period % 2 == 0:
                grad[:,idx + period * (size[-1] // periods)] = (float(idx) / (size[-1] / periods)) * peak
            else:
                grad[:,idx + period * (size[-1] // periods)] = peak - (float(idx) / (size[-1] / periods)) * peak
    grad += np.random.normal(0, (torch.max(grad) + torch.min(grad)) / 2)
    grad = Variable(grad)
    if flip:
        grad = grad.permute(1,0)
    if random.randint(0,1) == 0:
        grad = reverse_dim(grad, 0)
    if random.randint(0,1) == 0:
        grad = reverse_dim(grad, 1)
    if random.randint(0,1) == 0:
        grad = -grad
    if rotate:
        grad = rotate_and_scale(grad.unsqueeze(0).unsqueeze(0), None, 0.1)[0].squeeze()
    # return grad.cuda()
    return grad

def gen_tiles(size, dim=None, min_count=6, max_count=32, peak=0.5):
    assert len(size) == 2

    total_size = max(size)
    if dim is None:
        dim = random.randint(total_size // max_count, total_size // min_count) + 1

    count = total_size // dim + 1
    flip = half()
    tile = gen_gradient((dim,dim), flip=not flip, period_median=None, peak=peak, randomize_peak=True, rotate=False)
    tiles = tile.repeat(count,count)

    for idx in range(count-1):
        shift = random.randint(0, dim)
        shift_idxs = range(tiles.size(0))
        shift_idxs = list(shift_idxs[shift:]) + list(shift_idxs[:shift])
        # shift_idxs = torch.from_numpy(np.array(shift_idxs)).cuda().long()
        shift_idxs = torch.from_numpy(np.array(shift_idxs)).long()
        if flip:
            tiles[idx*dim:(idx+1)*dim,:] = tiles[idx*dim:(idx+1)*dim][:,shift_idxs]
        else:
            tiles[:,idx*dim:(idx+1)*dim] = tiles[shift_idxs][:,idx*dim:(idx+1)*dim]

    tiles = rotate_and_scale(tiles.unsqueeze(0).unsqueeze(0), np.pi/2, 0.2)[0].squeeze()
    tiles = tiles[:size[0],:size[1]]
    return tiles

def check_data_range(X, eps=1e-6, factor=1):
    mi, ma = torch.min(X).data[0], torch.max(X).data[0]
    assert mi >= -eps and ma <= factor + eps, 'Data must fall in range [0,1] ({}, {})'.format(mi, ma)

def aug_brightness(X, factor=2, mask=False, clamp=False):
    zm = X == 0

    total_periodic_augs = 2

    gradients = random.randint(0, total_periodic_augs)
    for _ in range(gradients):
        X = X + gen_gradient(X.size())
        if clamp:
            X = X.clamp(min=1/255.,max=1)

    tilings = random.randint(0, total_periodic_augs - gradients)
    for _ in range(tilings):
        X = X + gen_tiles(X.size())
        if clamp:
            X = X.clamp(min=1/255.,max=1)

    f = np.random.uniform(1,factor)
    X = X * half(f, 1./f)

    if not clamp:
        X = X + np.random.uniform(0,0.5)

    X[zm] = 0

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

def weighted_draw(l,h,exp=2, wf=None, max_factor=4):
    """
    Draw a value from the range [l,h] (inclusive) weighted such that
    the probability of drawing a value k is roughly proportional to the value
    itself (with an exponent to make the weighting less severe; exponent=0 gives
    a uniform distribution). The max factor caps the ratio max(weights) / min(weights)
    so that lower values don't become virtually non-existent.
    """
    weight_function = lambda x: (abs(x) - min(abs(l), abs(h)) + 1) ** float(exp) if wf is None else wf
    vals = range(l,h+1)
    weights = np.array([weight_function(v) for v in vals]).astype(np.float32)
    if max_factor is not None and max(weights) / min(weights) > max_factor:
        weights += (max(weights) - max_factor * min(weights)) / (max_factor - 1)
    weights /= np.sum(weights)
    return np.random.choice(vals, p=weights)

def random_rect_mask(size):
    dimx = random.randint(1,size[-2]//2)
    dimy = random.randint(1,size[-2]//2)
    mx = size[-2]//2-dimx//2
    my = size[-2]//2-dimy//2
    # prerotated_centered = Variable(torch.zeros(size)).cuda()
    prerotated_centered = torch.zeros(size)
    prerotated_centered[mx:mx+dimx,my:my+dimy] = 1
    rotated_centered, _ = rotate_and_scale(prerotated_centered, None, 0)
    upper_bound = int(size[-2]//2 - max(dimx,dimy) / np.sqrt(2))
    dx = weighted_draw(1,upper_bound,exp=1,max_factor=10)
    dy = weighted_draw(1,upper_bound,exp=1,max_factor=10)
    off_centered = Variable(translate(rotated_centered.data, dx, dy))
    output = torch.ceil(rotate_and_scale(off_centered, None, 0)[0])
    return output.byte()

def aug_input(x, factor=2):
    check_data_range(x)

    out = x
    zm = out == 0

    contrast_cutouts = half(0, random.randint(1,4))
    missing_cutouts = half(0, half(0,1))
    gaussian_cutouts = half(0, random.randint(1,2))
    missing_masks = []

    for _ in range(gaussian_cutouts):
        mask = random_rect_mask(out.size()).squeeze(0).squeeze(0)
        sigma = np.random.uniform(0.001,0.03)
        # noise = Variable(torch.FloatTensor(np.random.normal(0,sigma,out.size()))).cuda()
        noise = torch.FloatTensor(np.random.normal(0,sigma,out.size()))
        if half():
            # randomly smooth the noise
            r = random.randint(1,20)
            noise = noise.unsqueeze(0).unsqueeze(0)
            noise = F.avg_pool2d(noise, r*2+1, padding=r, stride=1, count_include_pad=False)
            noise = noise.view(out.size())
        out[mask] = out[mask] + noise[mask]

    for _ in range(contrast_cutouts):
        mask = random_rect_mask(out.size()).squeeze(0).squeeze(0)
        f = np.random.uniform(1,factor)
        out[mask] = out[mask] * half(f,1./f)

    for _ in range(missing_cutouts):
        mask = random_rect_mask(out.size()).squeeze(0)
        missing_masks.append(mask)
        mask = mask.squeeze(0)
        out[mask] = np.random.uniform(1/255.,1)

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
