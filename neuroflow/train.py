from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from model import Pyramid
from data import Data
import cavelab as cl
import time
from util import get_identity

if not torch.cuda.is_available():
    raise ValueError("Cuda is not available")

# improve loss
def mse_loss(input,target):
    #out = torch.pow((input[:,16:240,16:240]-target[:,16:240,16:240]), 2)
    out = torch.pow(input-target, 2)
    loss = out.sum()
    return loss


# Hierarchical training algorithm
# for level in mips.revers()
#   - load Data(level)
#   - load Gs = {G_i: i>level & i<level+5 & i<10}
#   - load Pyramid(Gs.locked())
#   - save Pyramid.G[0]

def train(hparams):
    # Load data
    d = Data(hparams)

    test_data = np.load('data/evaluate/simple_test.npy')

    # Load model
    input_shape = [hparams.batch_size, 2,256,256]
    model = Pyramid(levels = 5, shape=input_shape)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    model.train()

    for i in range(hparams.steps):
        _image, _target = d.get_batch()
        image = Variable(torch.from_numpy(_image).cuda())
        target = Variable(torch.from_numpy(_target).cuda())
        x = torch.stack([image, target], dim=1)

        optimizer.zero_grad()

        ys, Rs, rs = model(x)
        pred = torch.squeeze(ys[-1])

        loss = mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        if i%10 ==0:
            test(model, test_data)
            #cl.visual.save(pred.data.cpu().numpy()[0, :, :], 'dump/pred')
            #cl.visual.save(_image[0]/255, 'dump/image')
            #cl.visual.save(_target[0]/255, 'dump/target')
            print(i, 'loss', loss.data[0])
            torch.save(model, 'logs/MVP/model.pt')


def test(model, test_data):
    image = np.tile(test_data[:256,:256, 3]/256.0,(8,1,1)).astype(np.float32)
    target = np.tile(test_data[:256,:256, 4]/256.0,(8,1,1)).astype(np.float32)

    image = Variable(torch.from_numpy(image).cuda(),  requires_grad=False)
    target = Variable(torch.from_numpy(target).cuda(),  requires_grad=False)
    x = torch.stack([image, target], dim=1)

    ys, Rs, rs = model(x)

    #Draw
    y = ys[-1].data.cpu().numpy()[0]
    R = Rs[-1].data.cpu().numpy()[0]

    R = R - get_identity(batch_size=1, width=R.shape[-1])[0]
    R = np.transpose(R, (1,2,0))

    hsv, grid = cl.visual.flow(R)

    #### Draw
    cl.visual.save(y, 'dump/image')
    cl.visual.save(hsv, 'dump/target')
    cl.visual.save(grid, 'dump/pred')


if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    train(hparams)
