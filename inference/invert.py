import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
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
    print('Inverting fields in z_range {0}'.format(z_range))
    a.dst[0].add_composed_cv(block_start, inverse=False)
    a.dst[0].add_composed_cv(block_start, inverse=True)
    src_k = a.dst[0].get_composed_key(block_start, inverse=False) 
    dst_k = a.dst[0].get_composed_key(block_start, inverse=True) 
    src_cv = a.dst[0].for_read(src_k)
    dst_cv = a.dst[0].for_write(dst_k)
    for z in z_range:
      a.invert_field_chunkwise(z, src_cv, dst_cv, bbox, mip)
