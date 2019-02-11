import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--compose_start', help='compose_start', type=int)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.dst[0].add_composed_cv(args.compose_start, inverse=False)
  Fk = a.dst[0].get_composed_key(args.compose_start, inverse=False)
  # invFk = a.dst[0].get_composed_key(args.compose_start, inverse=True)
  field_cv = a.dst[0].for_read(Fk)
  
  dst_k = 'composed/render/F_{0}{1}'.format(args.dir_suffix, args.compose_start)
  path = join(args.dst_path, dst_k) 
  a.dst[0].add_path(dst_k, path, data_type='uint8', num_channels=1)
  a.dst[0].create_cv(dst_k)
  dst_cv = a.dst[0].for_write(dst_k)
  for z in z_range:
    a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)


