import torch
from torch import nn
from torch.nn.functional import conv2d, interpolate
from cloudvolume.lib import Bbox, Vec

import util
import math
import argparse

def create_gaussian_kernel(n=12, sigma=3):
  gx, gy = torch.meshgrid([torch.arange(n), torch.arange(n)])
  xy_grid = torch.stack([gx, gy], dim=-1).float()
  mean = (n-1)/2.
  var = sigma**2.
  gk = (1./(2.*math.pi*var)) * \
        torch.exp(-torch.sum((xy_grid-mean)**2., dim=-1) / (2.*var))
  gk = gk / torch.sum(gk)
  return gk.view(1, 1, n, n)

def blur(img, kernel_size, sigma):
  gk = create_gaussian_kernel(kernel_size, sigma)
  gk = gk.to(device=img.device, non_blocking=True)
  return conv2d(img, gk)

class Blur():

  def __init__(self, src_path, dst_path, src_mip, bbox_start, bbox_stop, 
               bbox_mip, kernel_size, sigma, disable_cuda):
    self.kernel_size = kernel_size
    self.sigma = sigma
    bbox = Bbox(bbox_start, bbox_stop)
    self.device = None
    if not disable_cuda and torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')
  
    self.src = util.get_cloudvolume(src_path, mip=src_mip)
    self.src_bbox = self.src.bbox_to_mip(bbox, bbox_mip, src_mip)
    self.dst = util.create_cloudvolume(dst_path, self.src.info, 
                                       src_mip, src_mip)
   
    self.pad = nn.ReplicationPad2d(self.kernel_size // 2)
    self.pad.to(device=self.device)
 
  def run(self):
    z_range = range(self.src_bbox.minpt[2], self.src_bbox.maxpt[2])
    for z in z_range:
      print('Blurring z={0}'.format(z))
      self.src_bbox.minpt[2] = z
      self.src_bbox.maxpt[2] = z+1
      dst_bbox = self.src_bbox
      print('src_bbox {0}'.format(self.src_bbox))
      # print('dst_bbox {0}'.format(dst_bbox))
      src_img = util.get_image(self.src, self.src_bbox)
      S = util.uint8_to_R(src_img)
      S = util.to_tensor(S, device=self.device).float()
      R = blur(self.pad(S), self.kernel_size, self.sigma)
      img = util.R_to_uint8(util.to_numpy(R))
      util.save_image(self.dst, dst_bbox, img)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
                    description='Create chunked pearson correlation image.')
  parser.add_argument('--src_path', type=str, 
    help='CloudVolume path of images to be evaluated')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path for where eval image written')
  parser.add_argument('--src_mip', type=int,
    help='MIP level of images to be used in evaluation')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--kernel_size', type=int, default=11,
    help='Gaussian kernel size')
  parser.add_argument('--sigma', type=int, default=2,
    help='Gaussian kernel standard deviation') 
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
  args = parser.parse_args()


