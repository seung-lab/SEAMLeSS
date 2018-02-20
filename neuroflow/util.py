import numpy as np
import cavelab as cl
import torch
import torchvision.utils as vutils

def get_identity(batch_size=8, width=256):
    identity = np.zeros((batch_size,2,width,width), dtype=np.float32)
    identity[:,0,:,:] = np.arange(width)/(width/2)-1
    identity[:,1,:,:] = np.transpose(identity, axes = [0,1,3,2])[:,0,:,:]
    return identity

def name(path):
    import socket
    from datetime import datetime
    import os
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(path, current_time)
    return log_dir

# Input four Pytorch variables that contain
def visualize(image, target, label, prediction, transform, n_iter, writer, name="Train/", crop=16):
    batch_size = image.shape[0]

    ### Vizualize examples
    im = vutils.make_grid(image[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    ta = vutils.make_grid(target[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    pr = vutils.make_grid(prediction[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)
    la = vutils.make_grid(label[:,crop:-crop,crop:-crop].data.unsqueeze(1), normalize=False, scale_each=True, nrow=4)

    writer.add_image(name+'Images/image', im, n_iter)
    writer.add_image(name+'Images/target', ta, n_iter)
    writer.add_image(name+'Images/labels', la, n_iter)
    writer.add_image(name+'Images/predictions', pr, n_iter)

    ### Optical Flow
    R = transform[:,:,crop:-crop,crop:-crop].data.cpu().numpy()
    R = R - get_identity(batch_size=batch_size, width=transform.shape[-1])[0][:,crop:-crop,crop:-crop]
    R = np.transpose(R, (0,2,3,1))

    width = transform.shape[-1]-2*crop
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
