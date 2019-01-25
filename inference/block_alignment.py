"""Serially align blocks of sections.

Each block's serial alignment will proceed as follows:
1. Copy the third section
2. Serially align the second to the third section without vector voting
3. Serially align the first section to the second section without vector voting
4. Serially align the fourth section through the end of the block using
   vector voting.
"""

import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--block_size', type=int, default=10)
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  # only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)

  mip = args.mip
  block_range = range(args.bbox_stop[2] // args.block_size)
  # create dict of dst cloudvolumes for each block
  dst_cv_dict = {}
  field_cv_dict = {}
  for block_start in block_range:
    dst_cv = a.dst[0].create(join(args.dst_path, 'serial_image', block_start), 
                             data_type='uint8', num_channels=1, fill_missing=True,
                             ignore_info=False, get_read=False)
    dst_cv_dict[block_start] = dst_cv

    field_cv = a.dst[0].create(join(args.dst_path, 'composed', block_start), 
                             data_type='uint8', num_channels=1, fill_missing=True,
                             ignore_info=False, get_read=False)
    field_cv_dict[block_start] = field_cv

  full_range = range(args.block_size)
  copy_range = full_range[2:3]
  pair_range = full_range[:2][::-1]
  vvote_range = full_range[3:-1]
  final_match = z_range[-1:]

  for offset in copy_range:
    for block_start in block_range:
      dst_cv = dst_cv_dict[block_start]
      z = block_start + offset
      a.copy_section(z, dst_cv, z, bbox, mip)

  # align without vector voting in the reverse direction
  a.tgt_radius = 1
  a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
  for offset in pair_range:
    for block_start in block_range:
      dst_cv = dst_cv_dict[block_start]
      field_cv = a.dst[-1].for_read('field')
      z = block_start + offset
      print('compute residuals without vector voting z={0}'.format(z))
      a.generate_pairwise([z], bbox, forward_match=False, reverse_match=True, 
                          render_match=False, batch_size=1)
      a.render(z, field_cv, z, dst_cv, z, bbox, a.render_low_mip)

  # align without vector voting in the reverse direction
  a.tgt_radius = args.tgt_radius
  a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
  for offset in vvote_range:
    for block_start in block_range:
      dst_cv = dst_cv_dict[block_start]
      field_cv = field_cv_dict[block_start]
      z = block_start + offset
      print('generate pairwise with vector voting z={0}'.format(z))
      a.generate_pairwise_and_compose([z], args.bbox_start[2], bbox, mip, 
                                      forward_match=True, reverse_match=False)
      a.render(z, field_cv, z, dst_cv, z, bbox, a.render_low_mip)

  a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)


