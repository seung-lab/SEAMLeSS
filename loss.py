import torch
import torch.nn as nn
from torch.autograd import Variable


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
    def penalty(fields, crack_mask, border_mask):
        if ptype ==     'lap': field = lap(fields)
        elif ptype == 'jacob': field = jacob(fields)
        elif ptype ==    'tv': field = tv(fields)
        else: crash # invalid penalty
        
        crack_mask = -nn.MaxPool2d(9,1,4)(-crack_mask) if crack_mask is not None else Variable(torch.ones(border_mask.size())).cuda()
        border_mask = nn.MaxPool2d(5,1,2)(border_mask)
        mask = (border_mask * crack_mask).view(field.size())
        if mask is not None:
            field = field * mask
        return torch.sum(field)
    return penalty
    
def similarity_score(should_reduce=False):
    return lambda x, y: torch.mean((x-y)**2) if should_reduce else (x-y)**2
