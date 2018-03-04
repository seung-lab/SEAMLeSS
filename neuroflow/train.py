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
from loss import loss
from util import get_identity, name, visualize
from tensorboardX import SummaryWriter
import json

if not torch.cuda.is_available():
    raise ValueError("Cuda is not available")

debug = True

def train(hparams):

    # Load Data, Model, Logging
    d = Data(hparams)
    path = os.path.join('logs/'+hparams.name, hparams.version)

    if not debug:
        writer = SummaryWriter(log_dir=path)
        with open(path+'/hparams.json', 'w') as outfile:
            json.dump(hparams, outfile)
    width = hparams.features['image']['width']
    input_shape = [hparams.batch_size, 2,width,width]
    model = Pyramid(levels = hparams.levels,
                    skip_levels=hparams.skip_levels,
                    shape=input_shape)

    model.cuda(device=0)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    model.train()
    image, target, lab = d.get_batch()
    image, target, lab = d.get_batch()
    new_image = np.zeros_like(image)
    width = 16
    new_image[0, :120, :] = image[0, :120,:]
    lab[0, 120:120+width, :] = 1
    new_image[0, 120+width:256,:] = image[0, 120:256-width,:]
    target = image
    image = new_image
    for i in range(hparams.steps):
        t1 = time.time()

        x = np.stack((image, target), axis=1)
        x = torch.autograd.Variable(torch.from_numpy(x).cuda(device=0), requires_grad=False)
        label = torch.autograd.Variable(torch.from_numpy(lab).cuda(device=0), requires_grad=False)

        optimizer.zero_grad()
        xs, ys, Rs, rs = model(x)

        level = int(i/20000)+1
        if level>len(xs):
            level = len(xs)
        level = 1
        l, mse, p1, norm = loss(xs[:level], ys[:level],
                                Rs[
                                :level], rs[:level], label,
                                start=0,
                                lambda_1=hparams.lambda_1,
                                lambda_2=hparams.lambda_2)
        l.backward(), optimizer.step()
        t2 = time.time()

        print(i, 'loss',  str(l.data[0])[:4],
                          str(p1.data[0])[:4],
                          str(t2-t1)[:4]+"s")

        #cl.visual.save(rs[j][0], 'dump/target')

        if i%hparams.log_iterations==0: # Takes 4s
            t3 = time.time()
            j = 0
            cl.visual.save(xs[j][0,0].data.cpu().numpy(), 'dump/image')
            cl.visual.save(ys[j][0,0].data.cpu().numpy(), 'dump/pred')
            cl.visual.save(xs[j][0,1].data.cpu().numpy(), 'dump/target')
            cl.visual.save(label[0].data.cpu().numpy(), 'dump/label')
            rss = np.transpose(Rs[j][0].data.cpu().numpy(), (1,2,0))
            #rss_2 = np.transpose(get_identity(batch_size=1, width=rss.shape[0])[0], (1,2,0))
            #rss_2 -= rss
            hsv, grid = cl.visual.flow(rss)
            cl.visual.save(hsv, 'dump/grid')
            for j in range(len(xs)):
                pass
                #visualize(xs[j][:8,0,:,:], xs[j][:8,1,:,:],
                #          label[:8, :, :], ys[j][:8,0,:,:],
                #          Rs[j][:8], rs[j][:8], rs[:(j+1)][:8],
                #          i, writer,
                #          mip_level=len(xs)-j-1,
                #          crop=(2**j+3))

            if not debug:
                writer.add_scalar('data/loss', l, i)
                writer.add_scalar('data/mse', mse, i)
                writer.add_scalar('data/p1', p1, i)
                writer.add_scalar('data/norm', norm, i)

                torch.save(model, path+'/model.pt') #0.3s
            t4 = time.time()
            print('visualize time', str(t4-t3)[:4]+"s")
    writer.close()

def test(model, test_data):
    image = np.tile((8,1,1)).astype(np.float32)
    target = np.tile(test_data[:256,:256, 4]/256.0,(8,1,1)).astype(np.float32)

    image = Variable(torch.from_numpy(image).cuda(),  requires_grad=False)
    target = Variable(torch.from_numpy(target).cuda(),  requires_grad=False)
    x = torch.stack([image, target], dim=1)

    ys, Rs, rs = model(x)

    vizuaize(image, target, ys[-1], Rs[-1], i, writer)


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

import sys
import os
from datetime import datetime
if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    myargs = getopts(sys.argv)
    if '--name' in myargs:  # Example usage.
        hparams.name = myargs['--name']
    if '--version' in myargs:
        hparams.version = myargs['--version']
    else:
        hparams.version = datetime.now().strftime('%b%d_%H-%M-%S')
    train(hparams)
