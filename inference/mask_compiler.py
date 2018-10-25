import torch
from torch import ByteTensor
from cloudvolume.lib import Bbox, Vec

import util
import math
import argparse

def logical_or(tensors):
  """Combine list of byte tensors with disjunction

  Args:
     tensors: list of tensors
  """
  assert(len(tensors) > 1)
  o = tensors[0] | tensors[1]
  for t in tensors[2:]:
    o = o | t
  return o

def logical_and(tensors):
  """Combine list of byte tensors with conjunction

  Args:
     tensors: list of tensors
  """
  assert(len(tensors) > 1)
  o = tensors[0] | tensors[1]
  for t in tensors[2:]:
    o = o & t
  return o

class MaskCompiler():

  def __init__(self, src_paths, dst_path, mip, bbox_start, bbox_stop, 
               bbox_mip, thresholds, disable_cuda, **kwargs):
    assert(len(src_paths) == len(thresholds))
    self.thresholds = thresholds
    bbox = Bbox(bbox_start, bbox_stop)
    self.device = None
    if not disable_cuda and torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')
 
    self.srcs = [util.get_cloudvolume(path, mip=mip) for path in src_paths]
    self.src_bbox = self.srcs[0].bbox_to_mip(bbox, bbox_mip, mip)
    self.dst = util.create_cloudvolume(dst_path, self.srcs[0].info, mip, mip)

  def get_bytetensor(self, src, bbox, threshold):
    img = util.get_image(src, bbox)
    S = util.uint8_to_R(img)
    S = util.to_tensor(S, device=self.device).float()
    return S >= threshold
   
  def run(self):
    print('src_bbox {0}'.format(self.src_bbox))
    print('src_path {0}'.format(self.srcs[0].path))
    print('threshold {0}'.format(self.thresholds[0]))
    M = self.get_bytetensor(self.srcs[0], self.src_bbox, self.thresholds[0])
    for src, th in zip(self.srcs[1:], self.thresholds[1:]):
      print('src_path {0}'.format(src.path))
      print('threshold {0}'.format(th))
      op = self.get_bytetensor(src, self.src_bbox, th)
      M = logical_or([M, op])
    mask = util.to_numpy(M)
    util.save_image(self.dst, self.src_bbox, mask)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
                    description='Threshold & combine CloudVolumes')
  parser.add_argument('--src_paths', nargs='+', type=str, 
    help='List of CloudVolume paths for images to be combined')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path for where combined image written')
  parser.add_argument('--mip', type=int,
    help='MIP level of images to be combined')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--thresholds', nargs='+', type=float,
    help='List of thresholds; length should match length of SRC_PATHS')
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
  args = parser.parse_args()

  m = MaskCompiler(**vars(args))
  m.run()

