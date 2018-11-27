import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--compose_start', 
    help='the earliest section to use in the composition',
    type=int) 
  parser.add_argument('--inverse_compose', 
    help='compute and store the inverse composition (aligning COMPOSE_START to Z)', 
    action='store_true')
  parser.add_argument('--forward_compose', 
    help='compute and store the forward composition (aligning Z to COMPOSE_START)', 
    action='store_true')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.compose_pairwise(z_range, args.compose_start, bbox, mip,
                     forward_compose=args.forward_compose,
                     inverse_compose=args.inverse_compose)
