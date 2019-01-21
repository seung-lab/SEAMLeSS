import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox

if __name__ == '__main__':
  parser = get_argparser()
  # parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--sigma', help='std of the bump function', type=float, default=1.4)

  args = parse_args(parser)
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  mip = args.mip

  z_start = args.bbox_start[2]
  z_stop = args.bbox_stop[2]

  # if args.concurrent_render:
  dst_k = 'field'
  path = join(args.dst_path, dst_k)
  a.dst[0].add_path(dst_k, path, data_type='float32', num_channels=2)
  a.dst[0].create_cv(dst_k)
  F_cv = a.dst[0].for_read(dst_k)

  dst_k = 'processed_field'
  path = join(args.dst_path, dst_k)
  a.dst[0].add_path(dst_k, path, data_type='float32', num_channels=2)
  a.dst[0].create_cv(dst_k)
  invF_cv = a.dst[0].for_write(dst_k)

  for z in range(z_start, z_stop):
    a.invert_field_chunkwise(z, F_cv, invF_cv, bbox, mip, optimizer=False)
