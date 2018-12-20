import torch
from torch import matmul, pow
from torch.nn.functional import softmax
from cloudvolume.lib import Bbox, Vec

import util
import argparse

def dist(U, V):
  D = U - V
  N = pow(D, 2)
  return pow(torch.sum(N, 3), 0.5).unsqueeze(0)

def get_diffs(fields):
  diffs = []
  for i in range(len(fields)):
    for j in range(i+1, len(fields)):
      diffs.append(dist(fields[i], fields[j])
  return torch.cat(diffs, dim=0)

def weight_diffs(diffs, T=1):
  return softmax(-diffs / T, dim=0)

def compile_field_weights(W):
  m = W.shape[0]
  n = int((1 + math.sqrt(1 + 8*m)) / 2)
  C = torch.zeros((n,) +  W.shape[1:])
  k = 0
  for i in range(n):
    for j in range(i+1, n):
      C[i,...] += W[k,...]
      C[j,...] += W[k,...]
      k += 1
  return C / (n-1)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
                    description='Create chunked pearson correlation image.')
  parser.add_argument('--field_paths', type=str, nargs='+',
    help='List of CloudVolume paths to images')
  parser.add_argument('--weight_path', type=str,
    help='CloudVolume path where weights written')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path where output image written')
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

  srcs = []
  for path in args.field_paths:
    srcs.append(util.get_field_cloudvolume(path, mip=args.mip))
  dst = util.create_field_cloudvolume(args.dst_path, srcs[0][0].info,
                                     args.mip, args.mip)
  wts = util.get_cloudvolume(args.weight_path, mip=args.mip)

  bbox = srcs[0][0].bbox_to_mip(bbox, args.bbox_mip, args.mip)
  fields = util.get_field(srcs[0], bbox)
  for src in srcs[1:]:
    u = util.get_field(src, bbox)
    fields = torch.cat([fields, u], dim=0)
  fields = fields.permute(1,2,0,3)
  w = util.to_tensor(util.get_image(wts, bbox))
  w = w.permute(3,2,1,0)
  v = matmul(w, fields)
  v = v.permute(2,0,1,3)
  util.save_field(dst, bbox, util.field_to_numpy(v))

