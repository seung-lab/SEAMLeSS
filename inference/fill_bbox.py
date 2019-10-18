from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
import numpy as np
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
  parser.add_argument('--fill_value', type=int, default=0,
    help='value to fill in the bbox')
  args = parse_args(parser)
  
  # Simplify var names
  mip = args.mip

  cv = CloudVolume(args.path, mip=mip, fill_missing=True, cdn_cache=False, non_aligned_writes=True)
  bbox0 = Bbox(args.bbox_start, args.bbox_stop)
  bbox = cv.bbox_to_mip(bbox0, args.bbox_mip, mip)
  new_img = np.full(bbox.size3(), args.fill_value).astype(np.dtype(cv.dtype))
  cv[bbox.to_slices()] = new_img
