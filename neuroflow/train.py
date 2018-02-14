from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils

from model import Pyramid
from data import Data
import cavelab as cl
import time
from util import get_identity, name
from tensorboardX import SummaryWriter

if not torch.cuda.is_available():
    raise ValueError("Cuda is not available")


# improve loss
def mse_loss(input,target):
    out = torch.pow((input[:,16:240,16:240]-target[:,16:240,16:240]), 2)
    #out = torch.pow(input-target, 2)
    loss = out.sum()
    return loss


# Hierarchical training algorithm
# for level in mips.revers()
#   - load Data(level)
#   - load Gs = {G_i: i>level & i<level+5 & i<10}
#   - load Pyramid(Gs.locked())
#   - save Pyramid.G[0]

def train(hparams):

    # Load Data, Model, Logging
    d = Data(hparams)
    test_data = np.load('data/evaluate/simple_test.npy')
    path = name('logs/'+hparams.name)
    writer = SummaryWriter(log_dir=path)

    input_shape = [hparams.batch_size, 2,256,256]
    model = Pyramid(levels = 5, shape=input_shape)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    model.train()

    for i in range(hparams.steps):
        _target, _image = d.get_batch()
        image = Variable(torch.from_numpy(_image).cuda())
        target = Variable(torch.from_numpy(_target).cuda())
        x = torch.stack([image, target], dim=1)

        optimizer.zero_grad()

        ys, Rs, rs = model(x)

        pred = torch.squeeze(ys[-1])
        loss = mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        if i%50 ==0: # Takes 4s
            #test(model, test_data) #0.7s
            print(i, 'loss', loss.data[0])

            vizuaize(image, target, pred, Rs[-1], i, writer)
            writer.add_scalar('data/loss', loss, i)
            torch.save(model, path+'/model.pt') #0.3s
    writer.close()

def test(model, test_data):
    #test_data[:256,:256].shape
    image = np.tile((8,1,1)).astype(np.float32)
    target = np.tile(test_data[:256,:256, 4]/256.0,(8,1,1)).astype(np.float32)

    image = Variable(torch.from_numpy(image).cuda(),  requires_grad=False)
    target = Variable(torch.from_numpy(target).cuda(),  requires_grad=False)
    x = torch.stack([image, target], dim=1)

    ys, Rs, rs = model(x)

    #Draw
    #y = ys[-1].data.cpu().numpy()[0]
    #R = Rs[-1].data.cpu().numpy()[0]

    #R = R - get_identity(batch_size=1, width=R.shape[-1])[0]
    #R = np.transpose(R, (1,2,0))

    #hsv, grid = cl.visual.flow(R)

    #### Draw
    vizuaize(image, target, ys[-1], Rs[-1], i, writer) # Takes 2s
    #cl.visual.save(y, 'dump/image')
    #cl.visual.save(hsv, 'dump/target')
    #cl.visual.save(grid, 'dump/pred')



# Input four Pytorch variables that contain
def vizuaize(image, target, prediction, transform, n_iter, writer, name="Train/"):
    batch_size = image.shape[0]

    ### Vizualize examples
    im = vutils.make_grid(image.data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    ta = vutils.make_grid(target.data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    pr = vutils.make_grid(prediction.data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)

    writer.add_image(name+'Images/image', im, n_iter)
    writer.add_image(name+'Images/target', ta, n_iter)
    writer.add_image(name+'Images/predictions', pr, n_iter)

    ### Optical Flow
    R = transform.data.cpu().numpy()
    R = R - get_identity(batch_size=batch_size, width=R.shape[-1])[0]
    R = np.transpose(R, (0,2,3,1))

    width = transform.shape[-1]
    hsvs = np.zeros((batch_size, 3, width, width))
    grids = np.zeros((batch_size, 3, 242, 242))

    for i in range(batch_size):
        hsv, grid = cl.visual.flow(R[i])
        hsvs[i,:,:,:] = np.transpose(hsv, (2,0,1))
        grids[i,:,:,:] = np.transpose(grid, (2,0,1))

    hsvs = vutils.make_grid(torch.from_numpy(hsvs), normalize=True, scale_each=True, nrow=4)
    grids = vutils.make_grid(torch.from_numpy(grids), normalize=True, scale_each=True, nrow=4)

    writer.add_image(name+'Images/flow', hsvs, n_iter)
    writer.add_image(name+'Images/field', grids, n_iter)


if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    train(hparams)
