import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--align_start',
    help='align without vector voting the 2nd & 3rd sections, otherwise copy them', action='store_true')
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  # only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)

  block_start = args.bbox_start[2]
  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.dst[0].add_composed_cv(block_start, inverse=False)
  field_k = a.dst[0].get_composed_key(block_start, inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  dst_cv = a.dst[0].for_write('dst_img')
  z_offset = 1
  uncomposed_field_cv = a.dst[z_offset].for_read('field')

  mip = args.mip
  copy_range = []
  uncomposed_range = []
  overlap_range = []
  composed_range = z_range
  if args.align_start:
    overlap = args.tgt_radius
    copy_range = z_range[0:1]
    uncomposed_range = z_range[1:overlap]
    overlap_range = z_range[overlap:2*overlap]
    composed_range = z_range[2*overlap:]

  # copy first section
  for z in copy_range:
    print('Copying z={0}'.format(z))
    a.copy_section(z, dst_cv, z, bbox, mip)
    a.downsample(dst_cv, z, bbox, a.render_low_mip, a.render_high_mip)

  # align without vector voting
  for z in uncomposed_range:
    print('Aligning without vector voting z={0}'.format(z))
    src_z = z
    tgt_z = z-1
    a.compute_section_pair_residuals(src_z, tgt_z, bbox)
    a.render_section_all_mips(src_z, uncomposed_field_cv, src_z,
                              dst_cv, src_z, bbox, mip)

  # align overlap to the uncomposed
  print('align overlap pairwise to previous aligned')
  for k, z in enumerate(overlap_range):
    for z_offset in range(k+1, args.tgt_radius+1):
      src_z = z
      tgt_z = src_z - z_offset
      # print(src_z,tgt_z)
      a.compute_section_pair_residuals(src_z, tgt_z, bbox)
    
  # setup an aligner to run pairwise 
  args.tgt_path = args.src_path 
  args.serial_operation = False 
  a = get_aligner(args)

  # align overlap to the composed
  print('align overlap pairwise to previous unaligned')
  for k, z in enumerate(overlap_range):
    for z_offset in range(1, k+1):
      src_z = z
      tgt_z = src_z - z_offset
      # print(src_z,tgt_z)
      a.compute_section_pair_residuals(src_z, tgt_z, bbox)
  print('align overlap pairwise to future unaligned')
  for k, z in enumerate(overlap_range):
    for z_offset in range(-args.tgt_radius, 0):
      src_z = z
      tgt_z = src_z - z_offset
      # print(src_z,tgt_z)
      a.compute_section_pair_residuals(src_z, tgt_z, bbox)

  # multi-match for all of the composed 
  a.generate_pairwise(composed_range, bbox, render_match=False)

  pairwise_range = list(overlap_range) + list(composed_range) 
  # compose from block_start) 
  a.compose_pairwise(pairwise_range, block_start, bbox, mip,
                     forward_compose=True,
                     inverse_compose=False)

  # render all of overlap and compose
  for z in pairwise_range:
    a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)

  # # compose and regularized all of the composed
  # for block_start in range(z_start, z_stop, args.block_size):
  #   compose_range = range(block_start, block_start + args.block_size + overlap)
  #   print('Composing for z_range {0}'.format(compose_range))
  #   a.compose_pairwise(compose_range, block_start, bbox, mip,
  #                      forward_compose=args.forward_compose,
  #                      inverse_compose=args.inverse_compose)

  #   reg_range = range(block_start, block_start + args.block_size)
  #   print('Regularizing for z_range {0}'.format(reg_range))
  #   a.regularize_z_chunkwise(reg_range, z_start, bbox, mip, sigma=args.sigma)
  #   for z in reg_range:
  #     a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)

