from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
import numpy as np
from copy import deepcopy
from args import get_argparser, parse_args 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--src_z', type=int)
  parser.add_argument('--dst_z_range', nargs='+', type=int)
  args = parse_args(parser)
  
  # Simplify var names
  mip = args.mip

  cv = CloudVolume(args.path, mip=mip, fill_missing=True, cdn_cache=False, non_aligned_writes=True)
  bbox0 = Bbox(args.bbox_start, args.bbox_stop)
  src_bbox = cv.bbox_to_mip(bbox0, args.bbox_mip, mip)
  src_bbox.minpt[2] = args.src_z
  src_bbox.maxpt[2] = args.src_z+1
  img = cv[src_bbox.to_slices()]
  dst_bbox = deepcopy(src_bbox)
  for dst_z in args.dst_z_range:
    print('Copying src_z={} to dst_z={}'.format(args.src_z, dst_z))
    dst_bbox.minpt[2] = dst_z
    dst_bbox.maxpt[2] = dst_z+1
    cv[dst_bbox.to_slices()] = img 
