import sys
import torch
from os.path import join
import numpy as np
from itertools import zip_longest, chain
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  # parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--single_field_path', type=str,
    help='CloudVolume path of single field to broadcast')
  parser.add_argument('--range_field_path', type=str,
    help='CloudVolume path of fields to be broadcasted over')
  parser.add_argument('--dst_field_path', type=str,
    help='CloudVolume path of where to store the resulting composed fields')
  parser.add_argument('--single_z', type=int)
  parser.add_argument('--batch_size', type=int, default=100)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  a.dst[0].add_path('single', args.single_field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].add_path('range', args.range_field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].add_path('dst_field', args.dst_field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].create_cv('single', ignore_info=True)
  a.dst[0].create_cv('range', ignore_info=True)
  a.dst[0].create_cv('dst_field', ignore_info=False)
  single_cv = a.dst[0].for_read('single')
  range_cv = a.dst[0].for_read('range')
  dst_cv = a.dst[0].for_write('dst_field')
  single_z = args.single_z
  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  mip = args.mip

  for z in z_range:
    a.compose_chunkwise(single_z, z, z, single_cv, range_cv, dst_cv, bbox,
                        mip, mip, mip) 
      
