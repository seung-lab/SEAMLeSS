import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--inverse_compose', 
    help='compute and store the inverse composition (aligning COMPOSE_START to Z)', 
    action='store_true')
  parser.add_argument('--forward_compose', 
    help='compute and store the forward composition (aligning Z to COMPOSE_START)', 
    action='store_true')
  # parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--sigma', help='std of the bump function', type=float, default=1.4)
  parser.add_argument('--compose_block',
    help='block size before using a new compose_start',
    type=int, default=7) 
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip
  overlap = args.tgt_radius
  if args.compose_block < 2*overlap + 1:
    args.compose_block = 2*overlap + 1

  for block_start in range(args.bbox_start[2], args.bbox_stop[2], args.compose_block-2*overlap):
    compose_range = range(block_start, block_start + args.compose_block)
    print('Composing for z_range {0}'.format(compose_range))
    a.compose_pairwise(compose_range, block_start, bbox, mip,
                       forward_compose=args.forward_compose,
                       inverse_compose=args.inverse_compose)

    reg_range = range(compose_range[0] + overlap, compose_range[-1] - overlap + 1)
    print('Regularizing for z_range {0}'.format(reg_range))
    a.regularize_z_chunkwise(reg_range, block_start, bbox, mip, sigma=args.sigma)
