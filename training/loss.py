import torch
import torch.nn as nn
from torch.autograd import Variable
from helpers import save_chunk
import numpy as np

def lap(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,:-2,:], p), 2)
    def dxf(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,2:,:,:], p), 1)
    def dyf(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,2:,:], p), 2)
    fields = map(lambda f: [dx(f), dy(f), dxf(f), dyf(f)], fields)        
    fields = map(lambda fl: (sum(fl) / 4.0) ** 2, fields)    
    field = sum(map(lambda f: torch.sum(f, -1), fields))
    return field

def jacob(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field

def cjacob(fields):
    def center(f):
        fmean_x, fmean_y = torch.mean(f[:,:,:,0]).data[0], torch.mean(f[:,:,:,1]).data[0]
        fmean = torch.cat((fmean_x * torch.ones((1,f.size(1), f.size(2),1)), fmean_y * torch.ones((1,f.size(1), f.size(2),1))), 3)
        fmean = Variable(fmean).cuda()
        return f - fmean
        
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        d = torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
        return center(d)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        d = torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
        return center(d)

    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field

def tv(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.abs(torch.cat(fields, -1)), -1)
    return field

def smoothness_penalty(ptype):
    def penalty(fields, weights=None):
        if ptype ==      'lap': field = lap(fields)
        elif ptype ==  'jacob': field = jacob(fields)
        elif ptype == 'cjacob': field = cjacob(fields)
        elif ptype ==     'tv': field = tv(fields)
        else: raise Exception('Smoothness penalty type {} not supported.'.format(ptype))
        
        if weights is not None:
            field = field * weights
        return field
    return penalty

def mse(x, y):
    return (x-y)**2

def similarity_penalty(ptype, should_reduce=False):
    def penalty(x, y, weights=None):
        if ptype == 'mse': field = mse(x, y)
        else: raise Exception('Similarity penalty type {} not supported.'.format(ptype))

        if weights is not None:
            field = field * weights

        return field
    return penalty
