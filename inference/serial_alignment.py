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

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.dst[0].add_composed_cv(args.bbox_start[2], inverse=False)
  field_k = a.dst[0].get_composed_key(args.bbox_start[2], inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  dst_cv = a.dst[0].for_write('dst_img')
  z_offset = 1
  uncomposed_field_cv = a.dst[z_offset].for_read('field')

  mip = args.mip
  composed_range = z_range[3:]
  if args.align_start:
    copy_range = z_range[0:1]
    uncomposed_range = z_range[1:3]
  else:
    copy_range = z_range[0:3]
    uncomposed_range = z_range[0:0]

  # copy first section
#  for z in copy_range:
#    print('Copying z={0}'.format(z))
#    a.copy_section(z, dst_cv, z, bbox, mip)
#    a.downsample(dst_cv, z, bbox, a.render_low_mip, a.render_high_mip)
#  # align without vector voting
#  for z in uncomposed_range:
#    print('compute residuals without vector voting z={0}'.format(z))
#    src_z = z
#    tgt_z = z-1
#    a.compute_section_pair_residuals(src_z, tgt_z, bbox)
#  #  a.render_section_all_mips(src_z, uncomposed_field_cv, src_z,
#  #                            dst_cv, src_z, bbox, mip) 
#    a.render(src_z, uncomposed_field_cv, src_z, dst_cv, src_z, bbox, a.render_low_mip)
#    a.downsample(dst_cv, src_z - 1, bbox, a.render_low_mip, a.render_high_mip)

#a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)
  # align with vector voting
  for z in composed_range:
      print('generate pairwise with vector voting z={0}'.format(z))
      a.generate_pairwise([z], bbox, render_match=False)
      print('compose pairwise with vector voting z={0}'.format(z))
      a.compose_pairwise([z], args.bbox_start[2], bbox, mip, forward_compose=True,
                         inverse_compose=False)
      src_z = z
      print('Aligning with vector voting z={0}'.format(z))
      a.render(src_z, field_cv, src_z, dst_cv, src_z, bbox, a.render_low_mip)
      print('Downsample z={0}'.format(z-1))
      a.downsample(dst_cv, src_z -1, bbox, a.render_low_mip, a.render_high_mip)

  print('Downsample z={0}'.format(composed_range[-1]))
  a.downsample(dst_cv, composed_range[-1], bbox, a.render_low_mip, a.render_high_mip)



  #vvblock = 20
  #for z_block in range(composed_range.start, composed_range.stop, vvblock):
  #    z_block_range = range(z_block, min(composed_range.stop, z_block + vvblock))
  #    print('generate pairwise with vector voting z={0}:{1}'.format(z_block_range.start,
  #                                                                  z_block_range.stop))
  #    a.generate_pairwise(z_block_range, bbox, render_match=False)
  #    print('compose pairwise with vector voting z={0}:{1}'.format(z_block_range.start,
  #                                                                 z_block_range.stop))
  #    a.compose_pairwise(z_block_range, args.bbox_start[2], bbox, mip,
  #                       forward_compose=True,
  #                       inverse_compose=False)
  #    for z in z_block_range:
  #        print('Aligning with vector voting z={0}'.format(z))
  #        src_z = z
  #        a.render(src_z, field_cv, src_z, dst_cv, src_z, bbox, a.render_low_mip)
  #        #a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)
  #    for z in z_block_range:
  #        print('Downsample z={0}'.format(z))
  #        src_z = z
  #        a.downsample(dst_cv, src_z, bbox, a.render_low_mip, a.render_high_mip)

