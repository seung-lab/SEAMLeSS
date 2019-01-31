import sys
import torch
import json
from time import time, sleep
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from tasks import run 

def print_run(diff, n_tasks):
  print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

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
  overlap = 3
  copy_range = range(args.block_size+overlap)
  broadcast_range = range(overlap-1, args.block_size+overlap)

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
                          fill_missing=True, overwrite=True)
  dst_field = cm.create(args.dst_field,
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  chunks = a.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                 cm.dst_voxel_offsets[mip], mip=mip, 
                                 max_mip=cm.num_scales)

  ###########################
  # Serial broadcast script #
  ###########################
  
  n_chunks = len(chunks) 
  # Copy vector field of first block
  batch = []
  block_start = block_range[0]
  prefix = block_start
  for block_offset in copy_range: 
    z = block_start + block_offset
    t = a.copy(cm, src_field, dst_field, z, z, bbox, mip, is_field=True, prefix=prefix)
    batch.extend(t)

  run(a, batch)
  start = time()
  # wait
  a.wait_for_queue_empty(dst_field.path, 'copy_done/{}'.format(prefix), len(batch))
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Render out the images from the copied field
  batch = []
  prefix = block_start
  for block_offset in copy_range: 
    z = block_start + block_offset 
    t = a.render(cm, src, dst_field, dst, src_z=z, field_z=z, dst_z=z, 
                 bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
    batch.extend(t)

  run(a, batch)
  start = time()
  # wait
  a.wait_for_queue_empty(dst.path, 'render_done/{}'.format(prefix), len(batch))
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Compose next block with last vector field from the previous composed block
  for block_start in block_range[1:]:
    batch = []
    z_broadcast = block_start + overlap - 1
    prefix = block_start
    for block_offset in broadcast_range[1:]:
      # decay the composition over the length of the block
      br = float(broadcast_range[-1])
      factor = (br - block_offset) / (br - broadcast_range[1])
      z = block_start + block_offset
      t = a.compose(cm, dst_field, src_field, dst_field, z_broadcast, z, z, 
                    bbox, mip, mip, mip, factor, prefix=prefix)
      batch.extend(t)

    run(a, batch)
    start = time()
    # wait
    a.wait_for_queue_empty(dst_field.path, 'compose_done/{}'.format(prefix), len(batch))
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    for block_offset in broadcast_range[1:]: 
      z = block_start + block_offset 
      t = a.render(cm, src, dst_field, dst, src_z=z, field_z=z, dst_z=z, 
                   bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
      batch.extend(t)

    run(a, batch)
    start = time()
    # wait 
    a.wait_for_queue_empty(dst.path, 'render_done/{}'.format(prefix), len(batch))
    end = time()
    diff = end - start
    print_run(diff, len(batch))


  # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)
