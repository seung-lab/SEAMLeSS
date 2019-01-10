import sys
import torch
import numpy as np
from cpc import CPC
from blur import Blur
from mask_compiler import MaskCompiler
import argparse

def get_dst_path(path, z_offset):
  return '{0}/z{1}'.format(path, z_offset)

def get_image_path(dst_path):
  return '{0}/image'.format(dst_path)

def get_cpc_path(dst_path, src_mip, dst_mip):
  return '{0}/cpc{1}{2}'.format(dst_path, src_mip, dst_mip)

def get_blur_path(dst_path, kernel_size, sigma):
  return '{0}/blur_{1}_{2}'.format(dst_path, kernel_size, sigma)

def get_mask_path(dst_path):
  return '{0}/mask'.format(dst_path)

def generate_masks(src_path, tgt_path, dst_dir, 
                    bbox_start, bbox_stop, bbox_mip, 
                    z_offsets, src_mip=4, dst_mip=8, 
                    kernel_size=5, sigma=1.1, min_r=0.2, 
                    r_slope=0.01, cache=False, **kwargs):
  """Generate masks based on blurred CPC comparison across z_range.

  Args:
     src_path: CloudVolume path to src image
     tgt_path: CloudVolume path to tgt image
     dst_dir: CloudVolume path to dst directory; root directory
       of dst/image, dst/cpc, dsts/blur, dst/mask
     bbox_start: starting coord of volume to be processed
     bbox_stop: stoppping coord of volume to be processed
     z_offsets: list of z_offsets to consider
     src_mip: cpc src MIP
     dst_mip: cpc dst MIP
     kernel_size: blur kernel width
     sigma: blur Gaussian std
     min_r: Pearson R threshold for z-1 neighbor
     r_slope: adjustment of Pearson R threshold based on z_offset
     cache: use CloudVolume caching
  """
  print('Generate masks')
  for z_offset in z_offsets:
    dst_path = get_dst_path(dst_dir, z_offset)
    cpc_path = get_cpc_path(dst_path, src_mip, dst_mip)
    blur_path = get_blur_path(cpc_path, kernel_size, sigma)
    print('src_path {0}'.format(src_path))
    print('tgt_path {0}'.format(tgt_path))
    print('dst_path {0}'.format(dst_path))
    print('blur_path {0}'.format(blur_path))
    e = CPC(src_path, tgt_path, cpc_path, src_mip, dst_mip, bbox_start,
             bbox_stop, bbox_mip, 0, z_offset, False, False, True)
    e.run()
    b = Blur(cpc_path, blur_path, dst_mip, bbox_start, bbox_stop, bbox_mip,
              kernel_size, sigma, False)
    b.run()

  # dst_paths = [get_dst_path(dst_dir, k) for k in z_offsets] 
  # cpc_paths = [get_cpc_path(path, src_mip, dst_mip) for path in dst_paths]
  # blur_paths = [get_blur_path(path, kernel_size, sigma) for path in cpc_paths]
  # mask_path = get_mask_path(dst_dir)
  # print('mask_path {0}'.format(mask_path))
  # thresholds = (min_r + r_slope) - r_slope*abs(np.array(z_offsets))
  # print('thresholds {0}'.format(thresholds))
  # m = MaskCompiler(blur_paths, mask_path, dst_mip, bbox_start, bbox_stop, 
  #                                                    0, thresholds, False)
  # m.run()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_path', type=str,
    help='CloudVolume path of images to be evaluated')
  parser.add_argument('--tgt_path', type=str,
    help='CloudVolume path of images to evaluate against; default: src_path')
  parser.add_argument('--dst_dir', type=str,
    help='Storage path where results will be stored (cpc & mask)')
  parser.add_argument('--src_mip', type=int, default=4,
    help='MIP level for CPC src_image')
  parser.add_argument('--dst_mip', type=int, default=8,
    help='MIP level for CPC output')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--z_offsets', nargs='+', type=int, default=[-1,-2,-3],
    help='Offset of each target z to compare against (e.g. -1, -2)')
  parser.add_argument('--kernel_size', type=int, default=5,
    help='Width of blur kernel')
  parser.add_argument('--sigma', type=float, default=1.1,
    help='Std of blur kernel Gaussian')
  parser.add_argument('--min_r', type=float, default=0.2,
    help='Minimum Pearson R accepted in mask for nearest z neighbor')
  parser.add_argument('--r_slope', type=float, default=0.01,
    help='Adjustment of Pearson R threshold based on z_offset')
  parser.add_argument('--cache', action='store_true', help='Use CloudVolume cache')
  args = parser.parse_args()
  
  generate_masks(**vars(args))
