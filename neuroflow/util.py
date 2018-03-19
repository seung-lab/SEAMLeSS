import numpy as np
import cavelab as cl
import torch
import torchvision.utils as vutils
import socket
from datetime import datetime
import os
import torch.nn.functional as F

def get_identity(batch_size=8, width=256):
    identity = np.zeros((batch_size,2,width,width), dtype=np.float32)+0.5
    identity[:,0,:,:] = np.arange(width)/((width-1)/2)-1
    identity[:,1,:,:] = np.transpose(identity, axes = [0,1,3,2])[:,0,:,:]
    return identity

def name(path, exp_name):

    log_dir = os.path.join(path, current_time)
    return log_dir

# Input four Pytorch variables that contain
def visualize(xs, xs_t, ys, Rs, rs, n_iter, writer, mip_level=0, label="", name="Train", crop=16):
    batch_size = xs.shape[0]
    name = name+'/'+str(mip_level)
    ### Vizualize examples
    im = vutils.make_grid(xs[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    ta = vutils.make_grid(xs_t[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    pr = vutils.make_grid(ys[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    #la = vutils.make_grid(label[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)

    writer.add_image(name+'/image', im, n_iter)
    writer.add_image(name+'/target', ta, n_iter)
    writer.add_image(name+'/predictions', pr, n_iter)

    R = np.transpose(Rs[:,:,crop:-crop,crop:-crop].data.cpu().numpy(), (0,2,3,1))
    r = np.transpose(rs[:,:,crop:-crop,crop:-crop].data.cpu().numpy(), (0,2,3,1))

    visualize_flow(R, writer, n_iter, name=name+'/R')
    visualize_flow(r, writer, n_iter, name=name+'/r')


def visualize_flow(flow, writer, n_iter, name=''):
    print(name, flow.shape)
    batch_size = flow.shape[0]
    width = flow.shape[1]
    hsvs = np.zeros((batch_size, 3, width, width))
    grids = np.zeros((batch_size, 3, 242, 242))

    for i in range(batch_size):
        hsv, grid = cl.visual.flow(flow[i])
        hsvs[i,:,:,:] = np.transpose(hsv, (2,0,1))
        grids[i,:,:,:] = np.transpose(grid, (2,0,1))

    hsvs = vutils.make_grid(torch.from_numpy(hsvs), normalize=True, scale_each=True, nrow=4)
    grids = vutils.make_grid(torch.from_numpy(grids), normalize=True, scale_each=True, nrow=4)

    writer.add_image(name+'/flow', hsvs, n_iter)
    writer.add_image(name+'/field', grids, n_iter)


def log_param_mean(model):
    levels = len(model.G_level)
    ls = {}
    for i in range(levels):
        mean = 0
        convs = list(filter(lambda x: isinstance(x, torch.nn.Conv2d), model.G_level[i].flow))
        for c in convs:
            mean += torch.mean(c.weight)
        mean /= len(convs)
        ls[str(i)] = mean.data.cpu().numpy()[0]
    return ls


def freeze(model, level=0):
    print('freeze', level)
    if level>len(model.G_level)-1:
        return
    for param in model.G_level[level].parameters():
        param.requires_grad = False

def freeze_all(model, besides=0):
    levels = len(model.G_level)
    for i in range(levels):
        for param in model.G_level[i].parameters():
            param.requires_grad = False

    for param in model.G_level[besides].parameters():
        param.requires_grad = True

def unfreeze(model):
    levels = len(model.G_level)
    for i in range(levels):
        for param in model.G_level[i].parameters():
            param.requires_grad = True
