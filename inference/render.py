import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  field_cv= a.dst[0].for_read('field')
  dst_cv = a.dst[0].for_write('dst_img')
  for z in z_range:
    a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)


