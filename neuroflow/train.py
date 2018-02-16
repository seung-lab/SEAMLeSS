from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import time
import numpy as np
import cavelab as cl
from model import Pyramid, G
from data import Data
from util import get_identity, name, vizuaize
from tensorboardX import SummaryWriter

if not torch.cuda.is_available():
    raise ValueError("Cuda is not available")

def mse_loss(input,target):
    out = torch.pow((input[:,16:240,16:240]-target[:,16:240,16:240]), 2)
    loss = out.mean()
    return loss

def smoothness_penalty(fields, order=1):
    factor = lambda f: f.size()[2] / 256
    dx =     lambda f: (f[:,:,1:,:] - f[:,:,:-1,:]) * factor(f)
    dy =     lambda f: (f[:,:,:,1:] - f[:,:,:,:-1]) * factor(f)
    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given k-th derivatives, compute (k+1)-th
    square_errors = map(lambda f: torch.sum(f ** 2, 1), fields) # sum along last axis (x/y channel)
    return sum(map(torch.mean, square_errors))

def loss(xs, ys, Rs, rs, lambd=0):
    s = smoothness_penalty([Rs[-1]], 2)
    pred = torch.squeeze(ys[-1])
    loss = mse_loss(pred, xs[-1][:,1,:,:])+lambd*s

    return loss

def train(hparams):

    # Load Data, Model, Logging
    d = Data(hparams)
    path = name('logs/'+hparams.name)
    writer = SummaryWriter(log_dir=path)

    input_shape = [hparams.batch_size, 2,256,256]
    model = Pyramid(levels = 5, shape=input_shape)
    model.cuda(device=0)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    model.train()

    for i in range(hparams.steps):
        t1 = time.time()
        _target, _image = d.get_batch()
        x = np.stack((_image, _target), axis=1)
        x = torch.autograd.Variable(torch.from_numpy(x).cuda(device=0))

        optimizer.zero_grad()
        xs, ys, Rs, rs = model(x)
        l = loss(xs, ys, Rs, rs, lambd=hparams.smooth_lambda)

        l.backward(), optimizer.step()
        t2 = time.time()
        print(i, 'loss', l.data[0], str(t2-t1)[:4]+"s")
        if i%hparams.log_iterations==0: # Takes 4s
            t3 = time.time()
            vizuaize(x[:8,0,:,:], x[:8,1,:,:], torch.squeeze(ys[-1][:8]), Rs[-1][:8], i, writer)
            writer.add_scalar('data/loss', l, i)
            torch.save(model, path+'/model.pt') #0.3s
            t4 = time.time()
            print('vizualize time', str(t4-t3)[:4]+"s")
    writer.close()

def test(model, test_data):
    image = np.tile((8,1,1)).astype(np.float32)
    target = np.tile(test_data[:256,:256, 4]/256.0,(8,1,1)).astype(np.float32)

    image = Variable(torch.from_numpy(image).cuda(),  requires_grad=False)
    target = Variable(torch.from_numpy(target).cuda(),  requires_grad=False)
    x = torch.stack([image, target], dim=1)

    ys, Rs, rs = model(x)

    vizuaize(image, target, ys[-1], Rs[-1], i, writer)


# Hierarchical training algorithm
# for level in mips.revers()
#   - load Data(level)
#   - load Gs = {G_i: i>level & i<level+5 & i<10}
#   - load Pyramid(Gs.locked())
#   - save Pyramid.G[0]

if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    train(hparams)
