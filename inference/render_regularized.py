import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--block_size',
    help='batch size for regularization; batches are necessary to prevent large vectors from accumulating during composition',
    type=int, default=10) 
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip
  overlap = args.tgt_radius
  if args.block_size < 2*overlap:
    args.compose_block = 2*overlap 
  z_start = args.bbox_start[2]
  z_stop = args.bbox_stop[2]

  # dst_cv = a.dst[0].for_write('dst_img')
  dst_k = 'image_regularized_{0}'.format(args.dir_suffix)
  path = join(args.dst_path, dst_k) 
  a.dst[0].add_path(dst_k, path, data_type='uint8', num_channels=1)
  a.dst[0].create_cv(dst_k)
  dst_cv = a.dst[0].for_write(dst_k)
  for block_start in range(z_start, z_stop, args.block_size):
    # regularized fields are saved in the next composed block
    next_cv = block_start + args.block_size
    a.dst[0].add_composed_cv(next_cv, inverse=False)
    field_cv = a.dst[0].get_composed_cv(next_cv, inverse=False, for_read=True)
    reg_range = range(block_start, block_start + args.block_size)
    print('Rendering for z_range {0}'.format(reg_range))
    for z in reg_range:
      a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)

