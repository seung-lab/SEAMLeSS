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
  parser.add_argument('--block_size',
    help='batch size for regularization; batches are necessary to prevent large vectors from accumulating during composition',
    type=int, default=10) 
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip
  overlap = args.tgt_radius
  if args.block_size < 2*overlap:
    args.block_size= 2*overlap 
  z_start = args.bbox_start[2]
  z_stop = args.bbox_stop[2]

  for block_start in range(z_start, z_stop, args.block_size - overlap):
    compose_range = range(block_start, block_start + args.block_size + overlap)
    print('Composing for z_range {0}'.format(compose_range))
    a.compose_pairwise(compose_range, block_start, bbox, mip,
                       forward_compose=args.forward_compose,
                       inverse_compose=args.inverse_compose)

    first_block = block_start == z_start
    reg_range = range(block_start, block_start + args.block_size)
    print('Regularizing for z_range {0}'.format(reg_range))
    a.regularize_z_chunkwise(reg_range, z_start, bbox, mip, sigma=args.sigma,
                             inverse=False, first_block=first_block)
    a.regularize_z_chunkwise(reg_range, z_start, bbox, mip, sigma=args.sigma,
                             inverse=True, first_block=first_block)
