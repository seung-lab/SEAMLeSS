from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from model import Pyramid
from data import Data
import cavelab as cl
import time

######################################################################
# Loading the data
# ----------------

test_data = np.load('data/evaluate/simple_test.npy')


######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.


use_cuda = torch.cuda.is_available()
batch_size = 2
input_shape = [batch_size, 2,256,256]
model = Pyramid(levels = 1, shape=input_shape)

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.00005)

def mse_loss(input,target):
    out = torch.pow((input[:,16:240,16:240]-target[:,16:240,16:240]), 2)
    loss = out.sum()
    return loss

def train(hparams):
    d = Data(hparams)

    model.train()
    #_image, _target = d.get_batch()
    _image, _target = np.load('data/trivial/image_fold.npy'), np.load('data/trivial/template_fold.npy')
    _image, _target = np.expand_dims(_image, axis=0), np.expand_dims(_target, axis=0)
    _image = np.repeat(_image, batch_size, axis=0).astype(np.float32)
    _target = np.repeat(_target, batch_size, axis=0).astype(np.float32)
    _image, _target = _image[:,128:128+256, 128:128+256], _target[:,128:128+256, 128:128+256]

    print(_image.shape)
    for i in range(hparams.steps):

        image = torch.from_numpy(_image)
        target = torch.from_numpy(_target)
        if use_cuda:
            image, target = image.cuda(), target.cuda()

        image, target = Variable(image), Variable(target)
        x = torch.stack([image, target], dim=1)
        optimizer.zero_grad()

        ys, Rs, rs = model(x)
        pred = torch.squeeze(ys[-1])

        loss = mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        if i%10 ==0:
            cl.visual.save(pred.data.cpu().numpy()[0, :, :], 'dump/pred')
            cl.visual.save(_image[0]/255, 'dump/image')
            cl.visual.save(_target[0]/255, 'dump/target')
            print(i, 'loss', loss.data[0])
            torch.save(model, 'logs/MVP/model.pt')


def test(model, test_data):
    y = np.zeros(test_data)
    for i in range(test_data.shape[2]-1)
        x = test_data[:,:,i:i+1]
        ys, Rs, rs = model(x)
        y[:,:,i] = ys[0]
    return y

if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    train(hparams)
