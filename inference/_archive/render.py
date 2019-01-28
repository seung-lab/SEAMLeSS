import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--field_path', type=str,
    help='CloudVolume path of field to use for render')
  parser.add_argument('--dst_image_path', type=str,
    help='CloudVolume path of rendered image')
  parser.add_argument('--ignore_info', 
    help='do not write over the info file in dst_image_path', 
    action='store_true')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  mip = args.mip

  a.dst[0].add_path('field', args.field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].create_cv('field', ignore_info=True)
  field_cv = a.dst[0].for_read('field')

  a.dst[0].add_path('final_img', args.dst_image_path, data_type='uint8', 
                  num_channels=1, fill_missing=True)
  a.dst[0].create_cv('final_img', ignore_info=args.ignore_info)
  dst_cv = a.dst[0].for_write('final_img')

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  for z in z_range:
    a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)

  a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)

