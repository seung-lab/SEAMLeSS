import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 
import csv
from itertools import zip_longest
from collections import OrderedDict
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--order_path', type=str)
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  # only compute matches to previous sections
  args.forward_matches_only = True
  a = get_aligner(args)
  bbox = get_bbox(args)

  vector_mip = args.mip
  image_mip = args.render_low_mip
  dst_cv = a.dst[0].for_write('dst_img_high_res')
  print("dst_cv", dst_cv)

  # determine order and field_cv from order_path
  range_list = OrderedDict()
  with open(args.order_path) as f:
    reader = csv.reader(f, delimiter=',')
    for k, r in enumerate(reader):
       if k != 0:
         z_start = int(r[0])
         z_stop = int(r[1])
         inc = int(r[2])
         compose_idx = int(r[3])
         if compose_idx not in range_list:
           range_list[compose_idx] = []
         range_list[compose_idx].append(range(z_start, z_stop+inc, inc))

  for compose_idx, ranges in range_list.items():
    z_pairs = zip_longest(*ranges)
    a.dst[0].add_composed_cv(compose_idx, inverse=False)
    field_k = a.dst[0].get_composed_key(compose_idx, inverse=False)
    field_cv= a.dst[0].for_read(field_k)

    for z_pair in z_pairs:
      z_pair = [z for z in z_pair if z]
      print((z_pair, compose_idx))
      for z in z_pair:  
        a.low_mip_render(z, field_cv, z, dst_cv, z, bbox, image_mip, vector_mip)
      if a.distributed:
        a.task_handler.wait_until_ready()
      a.downsample_range(dst_cv, z_pair, bbox, a.render_low_mip, a.render_high_mip)

