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
  def __init__(self, ndownsamples=4, currn=5, avgn=20, lambda1=0.4, 
                        lr=0.2, eps=0.01, min_iter=20, max_iter=1000):
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
      src = torch.FloatTensor(src).cuda()
      src = Variable(src).unsqueeze(0).unsqueeze(0)
      field = torch.FloatTensor(field).cuda()
      field = Variable(field).unsqueeze(0)
      #print(src, field)
      y =  F.grid_sample(src, field + self.get_identity_grid(field.size(2)))
      return  y.data.cpu().numpy()

  def process(self, src_image, dst_images, src_mask, dst_masks, crop=0):
    """Compute vector field that minimizes mean squared error (MSE) between 
    transformed src_image & all dst_images regularized by the smoothness of the 
    vector field subject to masks that allow the vector field to not be smooth. 
    The minimization uses SGD.

    Args:
    * src_image: nxm float64 ndarry normalized between [0,1]
      This is the image to be transformed by the returned vector field.
    * dst_images: list of nxm float64 ndarrays normalized between [0,1]
      This is the set of images that transformed src_image will be compared to.
    * src_mask: nxm float64 ndarray normalized between [0,1]
      The weight represents that degree to which a pixel participates in smooth
      deformation (0: not at all; 1: completely).
      1. This mask is used to reduce the smoothness penalty for pixels that 
      participate in a non-smooth deformation.
      2. This mask is transformed by the vector field and used to reduce the MSE
      for pixels that participate in a non-smooth deformation.
    * dst_mask: list of nxm float64 ndarrays normalized between [0,1]
      Exactly like the src_mask above. These masks are only used to reduce the 
      MSE for pixels that participate in a non-smooth deformation.
    
    Returns:
    * field: A nxmx2 float64 torch tensor normalized between [0,1]
      This field represents the derived vector field that transforms the 
      src_image subject to the contraints of the minimization.
    """
    print(src_image.shape, len(dst_images), src_mask.shape, len(dst_masks))
    downsample = lambda x: nn.AvgPool2d(2**x,2**x, count_include_pad=False) if x > 0 else (lambda y: y)
    upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    s = torch.FloatTensor(src_image)
    src_var = Variable((s - torch.mean(s)) / torch.std(s)).cuda().unsqueeze(0).unsqueeze(0)
    src_mask_var = Variable(torch.FloatTensor(src_mask)).cuda().unsqueeze(0).unsqueeze(0)
    dst_vars = []
    dst_mask_vars = []
    for d, m in zip(dst_images, dst_masks):
      d = torch.FloatTensor(d)
      dst_var = Variable((d - torch.mean(d)) / torch.std(d)).cuda().unsqueeze(0).unsqueeze(0)
      dst_vars.append(dst_var)
      mask_var = Variable(torch.FloatTensor(m)).cuda().unsqueeze(0).unsqueeze(0)
      dst_mask_vars.append(mask_var)
      
    dim = int(src_var.size()[-1] / (2 ** (self.ndownsamples - 1)))
    field = Variable(torch.zeros((1,dim,dim,2))).cuda().detach()
    field.requires_grad = True
    updates = 0
    for k in reversed(range(self.ndownsamples)):
      src_ = downsample(k)(src_var).detach()
      src_.requires_grad = False
      src_mask_ = downsample(k)(src_mask_var).detach()
      src_mask_.requires_grad = False
      dst_list_, dst_mask_list_ = [], []
      for d, m in zip(dst_vars, dst_mask_vars):
        dst_ = downsample(k)(d).detach()
        dst_.requires_grad = False
        dst_list_.append(dst_)
        mask_ = downsample(k)(m).detach()
        mask_.requires_grad = False
        dst_mask_list_.append(mask_)
      field = field.detach()
      field.requires_grad = True
      opt = torch.optim.SGD([field], lr=self.lr/(k+1))
      #sched = lr_scheduler.StepLR(opt, step_size=1, gamma=0.995)
      costs = []
      start_updates = updates
      print('Downsample factor {0}x'.format(2**k))
      while True:
        updates += 1
        I = self.get_identity_grid(field.size(2))
        pred = F.grid_sample(src_, field + I)
        pred_mask = F.grid_sample(src_mask_, field + I)
        centered_mask = self.center(src_mask_.squeeze(0), (1,2), 128/(2**k))
        centered_field = self.center(field, (1,2), 128/(2**k))
        penalty1 = self.penalty([centered_field], centered_mask)
        cost = penalty1 * self.lambda1/(k+1)
        for d, m in zip(dst_list_, dst_mask_list_):
          mask = torch.mul(pred_mask, m)
          mse = torch.mul(pred - d, mask)**2
          cost += torch.mean(self.center(mse, (-1,-2), 128 / (2**k)))
        #cost = diff + penalty1 * self.lambda1/(k+1)
        print(cost.data.cpu().numpy())
        costs.append(cost)
        cost.backward()
        opt.step()
        #sched.step()uniform
        opt.zero_grad()
        if len(costs) > self.avgn + self.currn and len(costs)>self.min_iter:
            hist_costs = costs[-(self.avgn+self.currn):-self.currn]
            hist = sum(hist_costs).data[0] / self.avgn
            curr = sum(costs[-self.currn:]).data[0] / self.currn
            if abs((hist-curr)/hist) < self.eps/(2**k) or len(costs)>self.max_iter:
                break
        #print downsamples, updates - start_updates
      if k > 0:
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
