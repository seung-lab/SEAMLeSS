import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
from os.path import join
from itertools import zip_longest, chain
import numpy as np
import time

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--center_range',
    help='indices of sections already aligned that will be extended in either direction',
    nargs='+', type=int)
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  # only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)

  a.dst[0].add_composed_cv(args.bbox_start[2], inverse=False)
  field_k = a.dst[0].get_composed_key(args.bbox_start[2], inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  dst_cv = a.dst[0].for_write('dst_img')

  positive_range = range(np.max(args.center_range)+1, args.bbox_stop[2]) 
  negative_range = range(np.min(args.center_range)-1, args.bbox_start[2]-1, -1) 
  z_range = chain(negative_range,  positive_range)

  mip = args.mip

  # align with vector voting
  for nz, pz in zip_longest(negative_range, positive_range):
    # redo work to give the other task enough time for object storage to sync
    if not nz:
      nz = negative_range[-1]
    if not pz:
      pz = positive_range[-1]
    print('generate pairwise with vector voting z={0}'.format(nz))
    a.generate_pairwise([nz], bbox, forward_match=False, 
                        reverse_match=True, render_match=False, 
                        batch_size=1, wait=False)
    print('generate pairwise with vector voting z={0}'.format(pz))
    a.generate_pairwise([pz], bbox, forward_match=True, 
                        reverse_match=False, render_match=False, 
                        batch_size=1, wait=True)
    print('compose pairwise with vector voting z={0}'.format(nz))
    a.compose_pairwise([nz], args.bbox_start[2], bbox, mip, forward_compose=True,
                       inverse_compose=False, negative_offsets=True,
                       serial_operation=True)
    print('compose pairwise with vector voting z={0}'.format(pz))
    a.compose_pairwise([pz], args.bbox_start[2], bbox, mip, forward_compose=True,
                       inverse_compose=False, negative_offsets=False,
                       serial_operation=True)
    a.task_handler.wait_until_ready()
    print('aligning with vector voting z={0}'.format(nz))
    a.render(nz, field_cv, nz, dst_cv, nz, bbox, a.render_low_mip, wait=False)
    print('aligning with vector voting z={0}'.format(pz))
    a.render(pz, field_cv, pz, dst_cv, pz, bbox, a.render_low_mip, wait=True)
    # a.downsample_range(dst_cv, [nz], bbox, a.render_low_mip, a.render_high_mip-2)
    # a.downsample_range(dst_cv, [pz], bbox, a.render_low_mip, a.render_high_mip-2)

  a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)


