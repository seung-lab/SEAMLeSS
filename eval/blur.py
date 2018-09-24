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

  bbox = Bbox(args.bbox_start, args.bbox_stop)
  args.device = None
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
  else:
    args.device = torch.device('cpu')

  src = util.get_cloudvolume(args.src_path, mip=args.src_mip)
  src_bbox = src.bbox_to_mip(bbox, args.bbox_mip, args.src_mip)
  dst = util.create_cloudvolume(args.dst_path, src.info, 
                                     args.src_mip, args.src_mip)
 
  pad = nn.ReplicationPad2d(args.kernel_size // 2)
  pad.to(device=args.device)

  for z in range(src_bbox.minpt[2], src_bbox.maxpt[2]):
    print('Blurring z={0}'.format(z))
    src_bbox.minpt[2] = z
    src_bbox.maxpt[2] = z+1
    dst_bbox = src_bbox
    print('src_bbox {0}'.format(src_bbox))
    # print('dst_bbox {0}'.format(dst_bbox))
    S = util.int8_to_norm(util.to_float(util.get_image(src, src_bbox)))
    S = util.to_tensor(S, device=args.device).float()
    R = blur(pad(S), args.kernel_size, args.sigma)
    img = util.to_uint8(util.norm_to_int8(util.to_numpy(R)))
    util.save_image(dst, dst_bbox, img)

