import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudvolume import CloudVolume
from cloudmanager import CloudManager

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--model_path', type=str,
    help='relative path to the ModelArchive to use for computing fields')
  parser.add_argument('--src_image_path', type=str)
  parser.add_argument('--pairwise_field_paths', type=json.loads,
    help='json dict of field paths indexed by offset')
  parser.add_argument('--final_field_path', type=str)
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
  parser.add_argument('--dst_image_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--max_displacement', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  args = parse_args(parser)
  # only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)

  info = CloudManager.create_info(CloudVolume(args.src_image_path), 
                                  args.max_mip, args.max_displacement)
  cm = CloudManager(info, provenance)
  src_cv = cm.create(args.src_image_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=True)
  dst_cv = cm.create(args.dst_image_path, data_type='uint8', 
                     num_channels=1, fill_missing=True, overwrite=True)
  src_mask_cv = None
  tgt_mask_cv = None
  if args.src_mask_path:
    src_mask_cv = cm.create(args.src_mask_path, data_type='uint8', num_channels=1,
                               fill_missing=True, overwrite=True)
    tgt_mask_cv = src_mask_cv

  field_cv_dict = {}
  for z_offset, field_path in args.pairwise_field_paths.items():
    field_cv_dict[z_offset] = cm.create(field_path, data_type='int16', num_channels=2,
                                        fill_missing=True, overwrite=True)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  mip = args.mip
  copy_range = z_range[2:3]
  uncomposed_range = z_range[0:2]
  composed_range = z_range[3:]

  # copy first section
  for z in copy_range:
    print('copying z={0}'.format(z))
    a.copy_chunkwise(cm, z, z, src_cv['read'], dst_cv['write'], bbox, mip)

  # align without vector voting
  for z in uncomposed_range:
    print('compute residuals without vector voting z={0}'.format(z))
    z_offset = -1
    src = src_cv['read']
    tgt = dst_cv['read']
    dst = dst_cv['write']
    field_w = field_cv_dict[z_offset]['write']
    field_r = field_cv_dict[z_offset]['read']
    a.compute_field_chunkwise(z, z-z_offset, src, tgt, field_w, bbox, mip)
    a.render_chunkwsie(z, field_r, z, dst, z, bbox, mip)

  # # align with vector voting
  # a.tgt_radius = args.tgt_radius
  # a.tgt_range = range(-a.tgt_radius, a.tgt_radius+1)
  # for z in composed_range:
  #   print('generate pairwise with vector voting z={0}'.format(z))
  #   a.generate_pairwise_and_compose([z], args.bbox_start[2], bbox, mip, 
  #                                   forward_match=True, reverse_match=False)
  #   print('aligning with vector voting z={0}'.format(z))
  #   a.render(z, field_cv, z, dst_cv, z, bbox, a.render_low_mip)

  # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)


