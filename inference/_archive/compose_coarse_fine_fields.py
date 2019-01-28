import sys
import torch
from os.path import join
import numpy as np
from itertools import zip_longest, chain
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  # parser.add_argument('--compose_start', help='earliest section composed', type=int)
  parser.add_argument('--coarse_field_path', type=str,
    help='CloudVolume path of coarse field')
  parser.add_argument('--fine_field_path', type=str,
    help='CloudVolume path of fine field')
  parser.add_argument('--dst_field_path', type=str,
    help='CloudVolume path of composed field')
  parser.add_argument('--coarse_mip', type=int)
  parser.add_argument('--fine_mip', type=int)
  parser.add_argument('--batch_size', type=int, default=100)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  a.dst[0].add_path('coarse', args.coarse_field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].add_path('fine', args.fine_field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].add_path('dst_field', args.dst_field_path, data_type='float32', 
                  num_channels=2, fill_missing=True)
  a.dst[0].create_cv('coarse', ignore_info=True)
  a.dst[0].create_cv('fine', ignore_info=True)
  a.dst[0].create_cv('dst_field', ignore_info=False)
  coarse_cv = a.dst[0].for_read('coarse')
  fine_cv = a.dst[0].for_read('fine')
  dst_field_cv = a.dst[0].for_write('dst_field')
  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  mip = args.mip

  k = 0  
  for z in z_range:
    a.compose_chunkwise(z, z, z, fine_cv, coarse_cv, dst_field_cv, bbox,
                        args.coarse_mip, args.fine_mip, args.fine_mip) 

    if k >= args.batch_size and a.distributed:
      print('wait')
      k = 0
      a.task_handler.wait_until_ready()
      
