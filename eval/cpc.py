import torch
from torch import pow, mul, reciprocal
from torch.nn.functional import conv2d, interpolate
from torch.nn import AvgPool2d, LPPool2d
from cloudvolume.lib import Bbox, Vec

import util
import argparse

def get_chunk_dim(scale_factor):
    return scale_factor, scale_factor

def center(X, scale_factor, device=torch.device('cpu')):
    chunk_dim = get_chunk_dim(scale_factor)
    avg_pool = AvgPool2d(chunk_dim, stride=chunk_dim).to(device=device)
    X_bar_down = avg_pool(X)
    X_bar = interpolate(X_bar_down, scale_factor=scale_factor, mode='nearest')
    return X - X_bar    

def cpc(S, T, scale_factor, device=torch.device('cpu')):
    chunk_dim = get_chunk_dim(scale_factor)
    sum_pool = LPPool2d(1, chunk_dim, stride=chunk_dim).to(device=device)
    S_hat = center(S, scale_factor, device=device)
    T_hat = center(T, scale_factor, device=device)
    S_hat_std = pow(sum_pool(pow(S_hat, 2)), 0.5)
    T_hat_std = pow(sum_pool(pow(T_hat, 2)), 0.5)
    norm = reciprocal(mul(S_hat_std, T_hat_std))
    R = mul(sum_pool(mul(S_hat, T_hat)), norm)
    return R

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
                    description='Create chunked pearson correlation image.')
  parser.add_argument('--src_path', type=str, 
    help='CloudVolume path of images to be evaluated')
  parser.add_argument('--tgt_path', type=str,
    help='CloudVolume path of images to compare against; default: src_path')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path for where eval image written')
  parser.add_argument('--src_mip', type=int,
    help='MIP level of images to be used in evaluation')
  parser.add_argument('--dst_mip', type=int,
    help='MIP level of output to be written. This dictates chunksize used.')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--composite_z', type=int, default=0,
    help='Number of z slices to create a composite image for scoring')
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
  args = parser.parse_args()

  args.tgt_path = args.tgt_path if args.tgt_path else args.src_path
  bbox = Bbox(args.bbox_start, args.bbox_stop)
  args.device = None
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
  else:
    args.device = torch.device('cpu')

  src = util.get_cloudvolume(args.src_path, mip=args.src_mip)
  tgt = util.get_cloudvolume(args.tgt_path, mip=args.src_mip)
  scale_factor = 2**(args.dst_mip - args.src_mip)
  dst_chunk = Vec(scale_factor, scale_factor, 1)
  src_bbox = src.bbox_to_mip(bbox, args.bbox_mip, args.src_mip)
  src_bbox = src_bbox.round_to_chunk_size(dst_chunk, offset=src.voxel_offset)
  dst = util.create_cloudvolume(args.dst_path, src.info, 
                                     args.src_mip, args.dst_mip)
  dst_bbox = dst.bbox_to_mip(src_bbox, args.src_mip, args.dst_mip) 
 
  for z in range(src_bbox.minpt[2], src_bbox.maxpt[2]):
    print('Scoring z={0}'.format(z))
    src_bbox.minpt[2] = z
    src_bbox.maxpt[2] = z+1
    if args.tgt_path != args.src_path:
      tgt_bbox = src_bbox
    else:
      tgt_bbox = Bbox(src_bbox.minpt+Vec(0,0,1), 
                      src_bbox.maxpt+Vec(0,0,args.composite_z+1))
    S = util.to_float(util.get_image(src, src_bbox))
    T = util.to_float(util.get_composite_image(tgt, tgt_bbox))
    S = util.to_tensor(S, device=args.device)
    T = util.to_tensor(T, device=args.device)
    R = cpc(S, T, scale_factor, device=args.device)
    img = util.to_uint8(util.adjust_range(util.to_numpy(R)))
    util.save_image(dst, dst_bbox, img)

