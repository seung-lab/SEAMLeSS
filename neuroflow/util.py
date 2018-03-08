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
    identity[:,0,:,:] = np.arange(width)/(width/2)-1
    identity[:,1,:,:] = np.transpose(identity, axes = [0,1,3,2])[:,0,:,:]
    return identity

def name(path, exp_name):

    log_dir = os.path.join(path, current_time)
    return log_dir

# Input four Pytorch variables that contain
def visualize(image, target, prediction, transform, res, rs, n_iter, writer, mip_level=0, label="", name="Train", crop=16):
    batch_size = image.shape[0]
    name = name+'/'+str(mip_level)
    ### Vizualize examples
    im = vutils.make_grid(image[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    ta = vutils.make_grid(target[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    pr = vutils.make_grid(prediction[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    #la = vutils.make_grid(label[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)

    writer.add_image(name+'/image', im, n_iter)
    writer.add_image(name+'/target', ta, n_iter)
    writer.add_image(name+'/predictions', pr, n_iter)

    R = transform[:,:,crop:-crop,crop:-crop].data.cpu().numpy()
    R = np.transpose(R, (0,2,3,1))

    r = np.transpose(res[:,:,crop:-crop,crop:-crop].data.cpu().numpy(), (0,2,3,1))

    visualize_flow(R, writer, n_iter, name=name+'/flow')
    visualize_flow(r, writer, n_iter, name=name+'/residual')


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
