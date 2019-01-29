"""Serially align blocks of sections.

Each block's serial alignment will proceed as follows:
1. Copy the third section
2. Serially align the second to the third section without vector voting
3. Serially align the first section to the second section without vector voting
4. Serially align the fourth section through the end of the block using
   vector voting.

Neighboring blocks will overlap by three sections. The last section of one block will be
the fixed (copied) section of the next block. The vector field from the last section
in the first block will be broadcast composed through the vector fields in the second
block, from the third section through to the final section.
"""
import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
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
  parser.add_argument('--max_displacement', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  args = parse_args(parser)
  # Only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.max_displacement

  # Compile ranges
  block_range = range(args.bbox_start[2], args.bbox_stop[2], args.block_size)
  overlap = 1 
  full_range = range(args.block_size + overlap)

  copy_range = full_range[:1]
  no_vvote_range = full_range[1:3]
  vvote_range = full_range[3:]

  no_vvote_offsets = [-1]
  vvote_offsets = [-3,-2,-1]

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  src_mask_cv = None
  tgt_mask_cv = None
  if args.src_mask_path:
    src_mask_cv = cm.create(args.src_mask_path, data_type='uint8', num_channels=1,
                               fill_missing=True, overwrite=True)
    tgt_mask_cv = src_mask_cv

  # Create dst CloudVolumes for each block, since blocks will overlap by 3 sections
  dst = cm.create(join(args.dst_path, 'image_blocks'),
                  data_type='uint8', num_channels=1, fill_missing=True, 
                  overwrite=True)

  # Create field CloudVolumes
  pair_fields = {}
  for z_offset in vvote_offsets:
    pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True)
  vvote_field = cm.create(join(args.dst_path, 'field', 'vvote'), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  # Copy first section
  for block_offset in copy_range:
    for block_start in block_range:
      z = block_start + block_offset 
      a.copy(cm, src, dst, z, z, bbox, mip, is_field=False, wait=False)

  # a.task_handler.wait()

  # Align without vector voting
  for block_offset in no_vvote_range:
    z_offset = no_vvote_offsets[0] 
    for block_start in block_range:
      z = block_start + block_offset 
      a.compute_field(cm, args.model_path, src, dst, vvote_field, 
                          z, z+z_offset, bbox, mip, pad, wait=False)
    # a.task_handler.wait()
    for block_start in block_range:
      z = block_start + block_offset 
      a.render(cm, src, vvote_field, dst, src_z=z, field_z=z, dst_z=z, 
                   bbox=bbox, src_mip=mip, field_mip=mip, wait=True)

  # Align with vector voting
  for block_offset in vvote_range:
    for block_start in block_range:
      z = block_start + block_offset 
      for z_offset in vvote_offsets:
        field = pair_fields[z_offset]
        a.compute_field(cm, args.model_path, src, dst, field, 
                            z, z+z_offset, bbox, mip, pad, wait=False)
    # a.task_handler.wait()
    for block_start in block_range:
      z = block_start + block_offset 
      a.vector_vote(cm, pair_fields, vvote_field, z, bbox, mip, 
                        inverse=False, softmin_temp=-1, serial=True, wait=False)
    # a.task_handler.wait()
    for block_start in block_range:
      z = block_start + block_offset 
      a.render(cm, src, vvote_field, dst, 
                   src_z=z, field_z=z, dst_z=z, 
                   bbox=bbox, src_mip=mip, field_mip=mip, wait=False)
    # a.task_handler.wait()

  # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)
