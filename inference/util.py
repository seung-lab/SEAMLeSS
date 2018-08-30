import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def crop(data_2d, crop):
    return data_2d[crop:-crop,crop:-crop]

def upsample_flow_(data_3d, factor):
    m = nn.Upsample(scale_factor=factor, mode='bilinear')
    data_4d = np.expand_dims(data_3d, axis=1)
    result = m(torch.from_numpy(data_4d))
    result = result.data.numpy()
    # account for shifting locations of -1 and +1 in upsampled flow
    result *= (data_4d.shape[2]-1.0)/data_4d.shape[2]
    result /= (result.shape[2]-1.0)/result.shape[2]
    return result[:, 0, :, :]

def upsample_flow(data_4d, factor):
    result = np.stack((upsample_flow_(data_4d[:, :, :, 0], factor),
                       upsample_flow_(data_4d[:, :, :, 1], factor)), axis=3)
    return result

def downsample_mip(data_3d):
    m = nn.AvgPool2d(2, stride=2)
    data_4d = np.expand_dims(data_3d, axis=1)
    result = m(Variable(torch.from_numpy(data_4d)))
    return result.data.numpy()[:, 0, :, :]

def warp(data, flow):
    td = torch.from_numpy(np.expand_dims(data, 0))
    tf = torch.from_numpy(flow)
    y = F.grid_sample(td, tf, mode='bilinear')
    return y.data.numpy()[0, 0]
