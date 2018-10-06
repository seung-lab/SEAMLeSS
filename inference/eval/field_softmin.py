import torch
from torch import pow
from torch.nn.functional import softmax
from cloudvolume.lib import Bbox, Vec

import util
import argparse

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
                    description='Create chunked pearson correlation image.')
  parser.add_argument('--paths', type=str, nargs='+', 
    help='List of CloudVolume paths to images')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path where output image written')
  parser.add_argument('--mip', type=int,
    help='MIP level of images to be used in evaluation')
  parser.add_argument('--temperature', type=int, default=-1,
    help='softmax temperature; default 2**MIP')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
  args = parser.parse_args()

  bbox = Bbox(args.bbox_start, args.bbox_stop)
  args.device = None
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
  else:
    args.device = torch.device('cpu')
  if args.temperature == -1:
    args.temperature = 2**args.mip

  srcs = []
  for path in args.paths:
    srcs.append(util.get_cloudvolume(path, mip=args.mip))
  dst = util.create_cloudvolume(args.dst_path, srcs[0].info, 
                                     args.mip, args.mip)
  dst.info['num_channels'] = 3
  dst.commit_info()
  
  bbox = srcs[0].bbox_to_mip(bbox, args.bbox_mip, args.mip)
  img = util.to_tensor(util.get_image(srcs[0] , bbox))
  for src in srcs[1:]:
    u = util.to_tensor(util.get_image(src, bbox))
    img = torch.cat([img, u], dim=0)
  F = softmax(-img / args.temperature, dim=0)
  # average softmax output for each field pair
  n = len(srcs)
  C = torch.zeros((n*(n-1)//2,) + F.shape[1:])
  k = 0
  for i in range(len(srcs)):
    for j in range(i+1, len(srcs)):
      C[k,...] = (F[i,...] + F[j,...]) / 2.
      k += 1
  util.save_image(dst, bbox, util.to_numpy(C))

