import torch
from cloudvolume.lib import Bbox, Vec
from cloudvolume import CloudVolume

import math
import util
import argparse

def create_field_bump(shape, sigma, device=torch.device('cuda')):
  """Create a 4D field bump
  
  Args:
     shape: 4-element tuple describing field shape (z, w, h, 2)
  """
  n = shape[0]
  mean = torch.tensor((n-1) / 2., device=device)
  var = torch.tensor(sigma**2., device=device)
  gz = torch.arange(n, dtype=torch.float, device=device)
  gk = (1./(2.*math.pi*var)) * torch.exp(-(gz-mean)**2. / (2.*var))
  gk = gk / torch.sum(gk)
  gk = gk.unsqueeze(1).unsqueeze(1).unsqueeze(1)
  return gk.repeat(1, *shape[1:])

def convolve(bump, field):
  """Convolve field bump with field
  """ 
  return torch.sum(torch.mul(bump, field), dim=0, keepdim=True)

def shift_field(cv, src_bbox, dst_bbox, field):
  """Remove & append z indices of field from CloudVolume

  Assume that src_bbox & dst_bbox match in xy, but not in z
  """
  remove_z = dst_bbox.minpt[2] - src_bbox.minpt[2] 
  new_bbox = Bbox(src_bbox.minpt + Vec(0,0,src_bbox.size3()[2]),
                  dst_bbox.maxpt)
  new_field = util.get_field(cv, new_bbox, device=field.device)
  return torch.cat(field[remove_z:,:,:,:], new_field, dim=0)

def bump_convolve(src, dst, bbox, bump_size=16, sigma=4, device=torch.device('cuda')):
  """Valid convolve 1D bump of BUMP_SIZE with CloudVolume SRC in BBOX and output DST
 
  Args
     src: CloudVolume of source field to convolve
     dst: CloudVolume where output field written 
     bbox: Bbox of region from SRC to convolve
     bump_size: int for size of 1D Gaussian (see create_field_bump) 
  """
  bump_dims = bump_size, bbox.size3()[0], bbox.size3()[1], 2
  bump = create_field_bump(bump_dims, sigma)
  f_size = Vec(bbox.size3()[0], bbox.size3()[1], bump_size)
  f_bbox = Bbox(bbox.minpt, bbox.minpt + f_size)
  o_size = Vec(bbox.size3()[0], bbox.size3()[1], 1)
  o_bbox = Bbox(bbox.minpt + Vec(0,0,bump_size//2), 
                bbox.minpt + Vec(0,0,bump_size//2) + o_size)
  field = util.get_field(src, f_bbox, device=device) 
  for i in range(bbox.size3()[2] - bump_size + 1):
    print('bump_convolve {0} to {1}'.format(f_bbox, o_bbox))
    o = convolve(bump, field)
    util.save_field(dst, o_bbox, util.field_to_numpy(o))
    if i < bbox.size3()[2] - bump_size:
      field = shift_field(src, f_bbox, f_bbox + Vec(0,0,1), field)
      f_bbox += Vec(0,0,1)
      o_bbox += Vec(0,0,1)
    
if __name__ == '__main__':

  parser = argparse.ArgumentParser(
              description='Combine vector fields based on voting.')
  parser.add_argument('--src_path', type=str,  
    help='CloudVolume path to input field')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path where output field written')
  parser.add_argument('--mip', type=int,
    help='MIP level of images to be used in evaluation')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--bump_size', type=int, default=16,
    help='Size of Gaussian bump (mean at size/2 with std of SIGMA)')
  parser.add_argument('--sigma', type=int, default=4,
    help='Standard deviation of Gaussian bump')
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
  args = parser.parse_args()

  bbox = Bbox(args.bbox_start, args.bbox_stop)
  args.device = None
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
  else:
    args.device = torch.device('cpu')

  src = util.get_cloudvolume(args.src_path, mip=args.mip)
  dst = util.create_cloudvolume(args.dst_path, src.info, 
                                     args.mip, args.mip)

  bbox = src.bbox_to_mip(bbox, args.bbox_mip, args.mip)
  bump_convolve(src, dst, bbox, bump_size=args.bump_size, sigma=args.sigma, device=args.device) 
