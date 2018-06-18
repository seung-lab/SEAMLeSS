import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

class Optimizer():
    def __init__(self, ndownsamples=4, currn=5, avgn=20, lambda1=0.4, lr=0.2, eps=0.01, min_iter=20, max_iter=1000):
        self.ndownsamples = ndownsamples
        self.currn = currn
        self.avgn = avgn
        self.lambda1 = lambda1
        self.lr = lr
        self.eps = eps
        self.identities = {}
        self.min_iter = min_iter
        self.max_iter = max_iter

    @staticmethod
    def center(var, dims, d):
        if not isinstance(d, collections.Sequence):
            d = [d for i in range(len(dims))]
        for idx, dim in enumerate(dims):
            if d[idx] == 0:
                continue
            var = var.narrow(dim, int(d[idx]/2), int(var.size()[dim] - d[idx]))
        return var

    def get_identity_grid(self, dim, cache=True):
        if dim not in self.identities:
            gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
            I = np.stack(np.meshgrid(gx, gy))
            I = np.expand_dims(I, 0)
            I = torch.FloatTensor(I)
            I = torch.autograd.Variable(I, requires_grad=False)
            I = I.permute(0,2,3,1).cuda()
            self.identities[dim] = I
        if cache:
            return self.identities[dim]
        else:
            return self.identities[dim].clone()

    def jacob(self, fields):
        def dx(f):
            p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
            return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
        def dy(f):
            p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
            return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
        field = torch.sum(torch.cat(fields, -1) ** 2, -1)
        return field

    def penalty(self, fields, mask=1):
        jacob = self.jacob(fields)
        jacob = torch.mul(jacob, mask)
        return torch.sum(jacob)

    def render(self, src, field):
        src, field = torch.FloatTensor(src).cuda(), torch.FloatTensor(field).cuda()
        src, field = Variable(src).unsqueeze(0).unsqueeze(0), Variable(field).unsqueeze(0)
        #print(src, field)
        y =  F.grid_sample(src, field + self.get_identity_grid(field.size(2)))
        return  y.data.cpu().numpy()

    def process(self, s, t, crop=0, mask=1):
        print(s.shape, t.shape)
        downsample = lambda x: nn.AvgPool2d(2**x,2**x, count_include_pad=False) if x > 0 else (lambda y: y)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        s, t = torch.FloatTensor(s), torch.FloatTensor(t)
        src = Variable((s - torch.mean(s)) / torch.std(s)).cuda().unsqueeze(0).unsqueeze(0)
        target = Variable((t - torch.mean(t)) / torch.std(t)).cuda().unsqueeze(0).unsqueeze(0)
        mask = Variable(torch.FloatTensor(mask)).cuda().unsqueeze(0)
        dim = int(src.size()[-1] / (2 ** (self.ndownsamples - 1)))
        field = Variable(torch.zeros((1,dim,dim,2))).cuda().detach()
        field.requires_grad = True
        updates = 0
        masking = not list(mask.shape)[-1] == 1
        for downsamples in reversed(range(self.ndownsamples)):
            src_, target_ = downsample(downsamples)(src).detach(), downsample(downsamples)(target).detach()
            mask_ = downsample(downsamples)(mask).detach() if masking else mask.detach()
            mask_.requires_grad = False
            src_.requires_grad = False
            target_.requires_grad = False
            field = field.detach()
            field.requires_grad = True
            opt = torch.optim.SGD([field], lr=self.lr/(downsamples+1))
            #sched = lr_scheduler.StepLR(opt, step_size=1, gamma=0.995)
            costs = []
            start_updates = updates
            print(downsamples)
            while True:
                updates += 1
                pred = F.grid_sample(src_, field + self.get_identity_grid(field.size(2)))
                if masking:
                    penalty1 = self.penalty([self.center(field, (1,2), 128 / (2**downsamples))], self.center(mask_, (1,2), 128 / (2**downsamples)))
                else:
                    penalty1 = self.penalty([self.center(field, (1,2), 128 / (2**downsamples))])
                diff = torch.mean(self.center((pred - target_)**2, (-1,-2), 128 / (2**downsamples)))
                cost = diff + penalty1 * self.lambda1/(downsamples+1)
                print(cost.data.cpu().numpy())
                costs.append(cost)
                cost.backward()
                opt.step()
                #sched.step()uniform
                opt.zero_grad()
                if len(costs) > self.avgn + self.currn and len(costs)>self.min_iter:
                    hist = sum(costs[-(self.avgn+self.currn):-self.currn]).data[0] / self.avgn
                    curr = sum(costs[-self.currn:]).data[0] / self.currn
                    if abs((hist-curr)/hist) < self.eps/(2**downsamples) or len(costs)>self.max_iter:
                        break
            #print downsamples, updates - start_updates
            if downsamples > 0:
                field = upsample(field.permute(0,3,1,2)).permute(0,2,3,1)
        #print(cost.data[0], diff.data[0], penalty1.data[0])
        print('done:', updates)
        print(field.shape)
        return self.center(field, (1,2), crop*2).data.cpu().numpy()[0]

if __name__ == '__main__':
    o = Optimizer()
    print('Testing...')
    s = np.random.uniform(0, 1, (256,256)).astype(np.float32)
    t = np.random.uniform(0, 1, (256,256)).astype(np.float32)

    flow = o.process(s, t)
    print(flow.shape)
    assert flow.shape == (1,256,256,2)

    flow = o.process(s, t, crop=10)
    assert flow.shape == (1,236,236,2)

    print ('All tests passed.')
