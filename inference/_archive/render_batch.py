import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--align_start', 
    help='align without vector voting the 2nd & 3rd sections, otherwise copy them', action='store_true')
  parser.add_argument('--compose_start', 
    help='the earliest section to use in the composition',
    type=int) 
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  # only compute matches to previous sections
  args.forward_matches_only = True
  a = get_aligner(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.dst[0].add_composed_cv(args.compose_start, inverse=False)
  field_k = a.dst[0].get_composed_key(args.compose_start, inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  #dst_cv = a.dst[0].for_write('dst_img_high_res')
  dst_cv = a.dst[0].for_write('dst_img')
  print("dst_cv", dst_cv)
  #z_offset = 1
  #uncomposed_field_cv = a.dst[z_offset].for_read('field')

  mip = args.mip
  for z in z_range:
    a.render_batch(z, field_cv, z, dst_cv, z, bbox, mip, 2)
    break

