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
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  args = parse_args(parser)
  # only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad
  copy_z = z_range[2]
  no_vvote_range = z_range[0:2][::-1]
  vvote_range = z_range[3:]

  info = CloudManager.create_info(CloudVolume(args.src_path), 
                                  max_mip, pad)
  cm = CloudManager(info, provenance)
  src_cv = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  dst_cv = cm.create(join(args.dst_path, 'image'), data_type='uint8', 
                     num_channels=1, fill_missing=True, overwrite=True)
  src_mask_cv = None
  tgt_mask_cv = None
  if args.src_mask_path:
    src_mask_cv = cm.create(args.src_mask_path, data_type='uint8', num_channels=1,
                               fill_missing=True, overwrite=True)
    tgt_mask_cv = src_mask_cv

  no_vvote_field_cv = cm.create(join(args.dst_path, 'field', str(1)), 
                                  data_type='int16', num_channels=2,
                                  fill_missing=True, overwrite=True)
  pairwise_field_cvs = {}
  pairwise_offsets = [-3,-2,-1]
  for z_offset in pairwise_offsets:
    pairwise_field_cvs[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                         data_type='int16', num_channels=2,
                                         fill_missing=True, overwrite=True)
  vvote_field_cv = cm.create(join(args.dst_path, 'field', 'vvote_{0}'.format(copy_z)), 
                                  data_type='int16', num_channels=2,
                                  fill_missing=True, overwrite=True)

  # copy first section
  print('copying z={0}'.format(copy_z))
  a.copy(cm, src_cv, dst_cv, copy_z, copy_z, bbox, mip, is_field=False, wait=True)

  # align without vector voting
  for z in no_vvote_range:
    print('compute residuals without vector voting z={0}'.format(z))
    a.compute_field(cm, args.model_path, src_cv, dst_cv, no_vvote_field_cv, 
                        z, z+1, bbox, mip, pad, wait=True)
    a.render(cm, src_cv, no_vvote_field_cv, dst_cv, 
                 src_z=z, field_z=z, dst_z=z, 
                 bbox=bbox, src_mip=mip, field_mip=mip, wait=True)

  # align with vector voting
  for z in vvote_range:
    for z_offset in pairwise_offsets:
      field_cv = pairwise_field_cvs[z_offset]
      a.compute_field(cm, args.model_path, src_cv, dst_cv, field_cv, 
                          z, z+z_offset, bbox, mip, pad, wait=False)
    a.vector_vote(cm, pairwise_field_cvs, vvote_field_cv, z, bbox, mip, 
                      inverse=False, softmin_temp=-1, serial=True, wait=True)
    a.render(cm, src_cv, vvote_field_cv, dst_cv, 
                 src_z=z, field_z=z, dst_z=z, 
                 bbox=bbox, src_mip=mip, field_mip=mip, wait=True)

  # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)


