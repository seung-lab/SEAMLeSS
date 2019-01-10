import torch
from torch import pow, mul, reciprocal
from torch.nn.functional import interpolate
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

class CPC():

  def __init__(self, src_path, tgt_path, dst_path, src_mip, dst_mip, 
            bbox_start, bbox_stop, bbox_mip, composite_z, z_offset, 
            forward_z, disable_cuda):
    self.src_mip = src_mip
    self.dst_mip = dst_mip
    self.forward_z = forward_z
    self.composite_z = abs(composite_z)
    self.z_offset = abs(z_offset)
    bbox = Bbox(bbox_start, bbox_stop)
    self.device = None
    if not disable_cuda and torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')
  
    self.src = util.get_cloudvolume(src_path, mip=src_mip)
    if tgt_path != src_path:
      self.tgt = util.get_cloudvolume(tgt_path, mip=src_mip)
    else:
      self.tgt = self.src
    self.scale_factor = 2**(dst_mip - src_mip)
    dst_chunk = Vec(self.scale_factor, self.scale_factor, 1)
    src_bbox = self.src.bbox_to_mip(bbox, bbox_mip, src_mip)
    self.src_bbox = src_bbox.round_to_chunk_size(dst_chunk, 
                                           offset=self.src.voxel_offset)
    self.dst = util.create_cloudvolume(dst_path, self.src.info, 
                                         src_mip, dst_mip)

  def run(self):
    z_range = range(self.src_bbox.minpt[2], self.src_bbox.maxpt[2])
    for z in z_range:
      print('Scoring z={0}'.format(z))
      self.src_bbox.minpt[2] = z
      self.src_bbox.maxpt[2] = z+1
      dst_bbox = self.dst.bbox_to_mip(self.src_bbox, self.src_mip, self.dst_mip) 
      tgt_adj = Vec(0,0,self.z_offset)
      if self.forward_z:
          min_adj = tgt_adj
          max_adj = Vec(0,0,self.composite_z) + tgt_adj
      else:
          min_adj = Vec(0,0,-self.composite_z) - tgt_adj
          max_adj = -tgt_adj
      tgt_bbox = Bbox(self.src_bbox.minpt + min_adj, 
                      self.src_bbox.maxpt + max_adj)
      print('src_bbox {0}'.format(self.src_bbox))
      S = util.to_float(util.get_image(self.src, self.src_bbox))
      print('tgt_bbox {0}'.format(tgt_bbox))
      T = util.to_float(util.get_composite_image(self.tgt, tgt_bbox, 
                                                 reverse=not self.forward_z))
      S = util.to_tensor(S, device=self.device)
      T = util.to_tensor(T, device=self.device)
      R = cpc(S, T, self.scale_factor, device=self.device)
      img = util.R_to_uint8(util.to_numpy(R))
      print('dst_bbox {0}'.format(dst_bbox))
      util.save_image(self.dst, dst_bbox, img)
    

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
    help='No. of z slices to create composite image')
  parser.add_argument('--z_offset', type=int, default=1,
    help='Offset in z for target slice')
  parser.add_argument('--forward_z', action='store_true',
    help='Create composite image from upcoming z indices')
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
  args = parser.parse_args()
  args.tgt_path = args.tgt_path if args.tgt_path else args.src_path


  r = CPC(**vars(args))
  r.run()

