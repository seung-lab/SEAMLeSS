import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox

if __name__ == '__main__':
  parser = get_argparser()
  # parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--sigma', help='std of the bump function', type=float, default=1.4)
  parser.add_argument('--block_size',
    help='batch size for regularization; batches are necessary to prevent large vectors from accumulating during composition',
    type=int, default=10)
  parser.add_argument('--concurrent_render',
    help='render after each block is regularized',
    action='store_true')
  args = parse_args(parser)
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  mip = args.mip
  overlap = args.tgt_radius
  if args.block_size < 2*overlap:
    args.block_size= 2*overlap
  z_start = args.bbox_start[2]
  z_stop = args.bbox_stop[2]

  # if args.concurrent_render:
  dst_k = 'image_regularized_{0}'.format(args.dir_suffix)
  path = join(args.dst_path, dst_k)
  a.dst[0].add_path(dst_k, path, data_type='uint8', num_channels=1)
  a.dst[0].create_cv(dst_k)
  dst_cv = a.dst[0].for_write(dst_k)

  for block_start in range(z_start, z_stop, args.block_size):
    compose_range = range(block_start, block_start + args.block_size + overlap)
    print('Composing for z_range {0}'.format(compose_range))
    a.compose_pairwise(compose_range, block_start, bbox, mip,
                       forward_compose=True, inverse_compose=False)
    if a.distributed:
      a.task_handler.wait_until_ready()

