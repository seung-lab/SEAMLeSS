import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--sigma', help='std of the bump function', type=float)
  parser.add_argument('--compose_block',
    help='block size before using a new compose_start',
    type=int, default=0) 
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip
  block_size = args.compose_block // 2
  # if args.compose_block == 0:
    # args.compose_block = args.bbox_stop[2] - args.bbox_start[2]
    # block_size = args.compose_block

  for block_start in range(args.bbox_start[2], args.bbox_stop[2], args.compose_block):
    z_range = range(block_start + block_size, block_start + block_size + args.compose_block)
    print('Regularizing for z_range {0}'.format(z_range))
    a.regularize_z_chunkwise(z_range, block_start, bbox, mip, sigma=args.sigma)
