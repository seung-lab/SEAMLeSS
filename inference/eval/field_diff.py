import torch
from torch import pow
from cloudvolume.lib import Bbox, Vec

import util
import argparse

def dist(U, V):
  D = U - V
  N = pow(D, 2)
  return pow(torch.sum(N, 3), 0.5).unsqueeze(0)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
                    description='Create chunked pearson correlation image.')
  parser.add_argument('--src_path', type=str, 
    help='CloudVolume path of images to be evaluated')
  parser.add_argument('--tgt_path', type=str,
    help='CloudVolume path of images to compare against')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path for where eval image written')
  parser.add_argument('--mip', type=int,
    help='MIP level of images to be used in evaluation')
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

  src = util.get_field_cloudvolume(args.src_path, mip=args.mip)
  tgt = util.get_field_cloudvolume(args.tgt_path, mip=args.mip)
  dst = util.create_cloudvolume(args.dst_path, src[0].info, 
                                     args.mip, args.mip)

  bbox = src[0].bbox_to_mip(bbox, args.bbox_mip, args.mip)
  U = util.get_field(src, bbox)
  V = util.get_field(tgt, bbox)
  D = dist(U, V)
  util.save_image(dst, bbox, util.diff_to_numpy(D))

