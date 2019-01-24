import sys
import torch
from os.path import join
import math
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--offset', type=int, default=1,
    help='offset of section to compare against')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  a.dst[0].add_path('final_img', args.dst_path, data_type='uint8', 
                  num_channels=1, fill_missing=True)
  a.dst[0].create_cv('final_img', ignore_info=False)
  src_cv = a.src['src_img']
  tgt_cv = a.src['src_img']
  dst_cv = a.dst[0].for_write('final_img')
  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  for z in z_range:
    a.cpc_chunkwise(z, z+args.offset, src_cv, tgt_cv, dst_cv, bbox,
                    args.render_low_mip, args.render_high_mip)
