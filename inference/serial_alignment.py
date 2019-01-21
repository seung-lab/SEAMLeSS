import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  # only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  # a.dst[0].add_path('dst_temp_img', join(args.dst_path, 'block_image'), 
  #                   data_type='uint8', num_channels=1, fill_missing=True)
  # a.dst[0].create_cv('dst_temp_img', ignore_info=False)
  dst_cv = a.dst[0].for_write('dst_img')
  a.dst[0].add_composed_cv(args.bbox_start[2], inverse=False)
  field_k = a.dst[0].get_composed_key(args.bbox_start[2], inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  z_offset = 1
  uncomposed_field_cv = a.dst[z_offset].for_read('field')

  mip = args.mip
  copy_range = z_range[0:1]
  uncomposed_range = z_range[1:3]
  composed_range = z_range[3:]

  # copy first section
  for z in copy_range:
    print('copying z={0}'.format(z))
    a.copy_section(z, dst_cv, z, bbox, mip)

  # align without vector voting
  for z in uncomposed_range:
    print('compute residuals without vector voting z={0}'.format(z))
    a.tgt_radius = 1
    a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
    a.generate_pairwise([z], bbox, forward_match=True, reverse_match=False, 
                        render_match=False, batch_size=1)
    a.render(z, uncomposed_field_cv, z, dst_cv, z, bbox, a.render_low_mip)

  # align with vector voting
  a.tgt_radius = args.tgt_radius
  a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
  for z in composed_range:
    print('generate pairwise with vector voting z={0}'.format(z))
    a.generate_pairwise_and_compose([z], args.bbox_start[2], bbox, mip, 
                                    forward_match=True, reverse_match=False)
    print('aligning with vector voting z={0}'.format(z))
    a.render(z, field_cv, z, dst_cv, z, bbox, a.render_low_mip)

  a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)


