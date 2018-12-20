import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--render_match', 
    help='render source with all single pairwise transforms before weighting',
    action='store_true')
  parser.add_argument('--compose_start', 
    help='the earliest section to use in the composition',
    type=int) 
  parser.add_argument('--inverse_compose', 
    help='compute and store the inverse composition (aligning COMPOSE_START to Z)', 
    action='store_true')
  parser.add_argument('--forward_compose', 
    help='compute and store the forward composition (aligning Z to COMPOSE_START)', 
    action='store_true')
  parser.add_argument('--sigma', help='std of the bump function', type=float)
  parser.add_argument('--block_size',
    help='batch size for regularization; batches are necessary to prevent large vectors from accumulating during composition',
    type=int, default=10) 

  args = parse_args(parser) 
  args.tgt_path = join(args.dst_path, 'image')
  a = get_aligner(args)
  bbox = get_bbox(args)
  forward_compose = args.forward_compose
  inverse_compose = args.inverse_compose 
  compose_start = args.bbox_start[2]  #args.compose_start
  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.listen_for_tasks(args.bbox_start[2], args.bbox_stop[2] - args.bbox_start[2], bbox, 
                     forward_compose, inverse_compose, compose_start)
