import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--src_field', type=str)
  parser.add_argument('--dst_field', type=str)
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
  broadcast_range = range(overlap-1, args.block_size+overlap)

  no_vvote_range = broadcast_range[1:3]
  vvote_range = broadcast_range[3:]

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
  dst = cm.create(args.dst_path, 
                  data_type='uint8', num_channels=1, fill_missing=True, 
                  overwrite=True)

  # Create field CloudVolumes
  src_field = cm.create(args.src_field,
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)
  dst_field = cm.create(args.dst_field,
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  # Copy vector field of first block
  block_start = block_range[0]
  for block_offset in broadcast_range: 
    z = block_start + block_offset
    print("Copy z={}".format(z))
    a.copy(cm, src_field, dst_field, z, z, bbox, mip, is_field=True, wait=False)
  # a.task_handler.wait()
  for block_offset in broadcast_range: 
    z = block_start + block_offset 
    print("Render z={}".format(z))
    a.render(cm, src, dst_field, dst, src_z=z, field_z=z, dst_z=z, 
                 bbox=bbox, src_mip=mip, field_mip=mip, wait=True)

  # Compose next block with last vector field from the previous composed block
  for block_start in block_range[1:]:
    z_broadcast = block_start + overlap - 1
    for block_offset in broadcast_range[1:]:
      z = block_start + block_offset
      print("Broadcast z={} over z={}".format(z_broadcast, z))
      a.compose(cm, dst_field, src_field, dst_field, z_broadcast, z, z, 
                bbox, mip, mip, mip, wait=False)
    # a.task_handler.wait()
    for block_offset in broadcast_range[1:]: 
      z = block_start + block_offset 
      print("Render z={}".format(z))
      a.render(cm, src, dst_field, dst, src_z=z, field_z=z, dst_z=z, 
                   bbox=bbox, src_mip=mip, field_mip=mip, wait=True)
    # a.task_handler.wait()

  # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)
