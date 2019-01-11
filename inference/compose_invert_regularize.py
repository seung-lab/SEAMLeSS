import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
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

  for block_start in range(z_start, z_stop, args.block_size):
    compose_range = range(block_start, block_start + args.block_size + overlap)
    print('Composing for z_range {0}'.format(compose_range))
    if block_start != z_start:
      a.compose_pairwise(compose_range, block_start, bbox, mip,
                       forward_compose=args.forward_compose,
                       inverse_compose=False)
    print('Inverting composed fields for z_range {0}'.format(compose_range))
    curr_block = compose_range[0]
    next_block = curr_block + args.block_size 
    a.dst[0].add_composed_cv(curr_block, inverse=False)
    a.dst[0].add_composed_cv(curr_block, inverse=True)
    F_cv = a.dst[0].get_composed_cv(curr_block, inverse=False, for_read=True)
    invF_cv = a.dst[0].get_composed_cv(curr_block, inverse=True, for_read=False)
    for z in compose_range:
      a.invert_field_chunkwise(z, F_cv, invF_cv, bbox, mip)

    reg_range = range(block_start, block_start + args.block_size)
    print('Regularizing for z_range {0}'.format(reg_range))
    a.regularize_z_chunkwise(reg_range, z_start, bbox, mip, sigma=args.sigma)
    # print('Inverting regularized fields for z_range {0}'.format(reg_range[-overlap:]))
    # a.dst[0].add_composed_cv(next_block, inverse=False)
    # a.dst[0].add_composed_cv(next_block, inverse=True)
    # F_cv = a.dst[0].get_composed_cv(next_block, inverse=False, for_read=True)
    # invF_cv = a.dst[0].get_composed_cv(next_block, inverse=True, for_read=False)
    # for z in reg_range[-overlap:]:
    #   a.invert_field_chunkwise(z, F_cv, invF_cv, bbox, mip)

