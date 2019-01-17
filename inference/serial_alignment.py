import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--no_vvote_start',
    help='copy 1st section & align the 2nd & 3rd without vector voting',
    action='store_true')
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
  print(dst_cv)
  z_offset = 1
  uncomposed_field_cv = a.dst[z_offset].for_read('field')

  image_mip = args.image_mip
  field_mip = args.field_mip
  if args.no_vvote_start:
    copy_range = z_range[0:1]
    uncomposed_range = z_range[1:3]
    composed_range = z_range[3:]
  else:
    copy_range = z_range[0:0]
    uncomposed_range = z_range[0:0]
    composed_range = z_range

  # copy first section
  for z in copy_range:
    print('copying z={0}'.format(z))
    a.copy_section(z, dst_cv, z, bbox, image_mip)

  # align without vector voting
  for z in uncomposed_range:
    print('compute residuals without vector voting z={0}'.format(z))
    a.tgt_radius = 1
    a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
    a.generate_pairwise([z], bbox, input_mip=image_mip, forward_match=True, reverse_match=False, 
                        render_match=False, batch_size=1)
    a.low_mip_render(z, uncomposed_field_cv, z, dst_cv, z, bbox, image_mip, field_mip)

  # align with vector voting
  a.tgt_radius = args.tgt_radius
  a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
  for z in composed_range:
      print('generate pairwise with vector voting z={0}'.format(z))
      a.generate_pairwise([z], bbox, input_mip=image_mip, forward_match=True, 
                          reverse_match=False, render_match=False)
      print('compose pairwise with vector voting z={0}'.format(z))
      a.compose_pairwise([z], args.bbox_start[2], bbox, field_mip, forward_compose=True,
                         inverse_compose=False, serial_operation=True)
      print('aligning with vector voting z={0}'.format(z))
      a.low_mip_render(z, field_cv, z, dst_cv, z, bbox, image_mip, field_mip)

  a.downsample_range(dst_cv, z_range, bbox, field_mip, a.render_high_mip)


