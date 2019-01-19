import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  # parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--field_path', type=str,
    help='CloudVolume path of vector field')
  parser.add_argument('--image_mip', type=int)
  parser.add_argument('--field_mip', type=int)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  a.dst[0].add_path('field', args.field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].create_cv('field', ignore_info=True)
  src_cv = a.src['src_img']
  field_cv = a.dst[0].for_read('field')
  dst_cv = a.dst[0].for_write('dst_img_high_res')
  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  a.upsample_render_rechunk(z_range, src_cv, field_cv, dst_cv, 
                              bbox, args.image_mip, args.field_mip) 
