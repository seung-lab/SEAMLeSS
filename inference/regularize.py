import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--sigma', help='std of the bump function', type=float)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.regularize_z_chunkwise(z_range, args.compose_start, bbox, mip, sigma=args.sigma)

