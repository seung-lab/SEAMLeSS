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
  parser.add_argument('--queue_name', type=str, default=None)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.listen_for_tasks(args.bbox_start[2], args.bbox_stop[2] - args.bbox_start[2], bbox)
