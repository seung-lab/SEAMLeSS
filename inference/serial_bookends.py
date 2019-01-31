"""Create serially aligned bookends ahead of a pairwise run
"""
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
  block_range = range(args.bbox_start[2], args.bbox_stop[2]+args.block_size, args.block_size)
  full_range = range(args.block_size)

  forward_reverse = [1,-1]
  copy_range = [0]
  serial_range = [-1,-2] 
  pair_range = [1,2,3]

  serial_offset = 1
  pair_offsets = {1: [-1,-2,-3],
                  2: [-2,-3],
                  3: [-3]}

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  src_mask_cv = None
  tgt_mask_cv = None
  if args.src_mask_path:
    src_mask_cv = cm.create(args.src_mask_path, data_type='uint8', num_channels=1,
                               fill_missing=True, overwrite=False)
    tgt_mask_cv = src_mask_cv

  # Create dst CloudVolumes for each block, since blocks will overlap by 3 sections
  dst = cm.create(join(args.dst_path, 'bookends'),
                  data_type='uint8', num_channels=1, fill_missing=True, 
                  overwrite=True)

  # Create field CloudVolumes
  serial_fields = {}
  for fr in forward_reverse:
    z_offset = serial_offset * fr
    serial_fields[z_offset] = cm.create(join(args.dst_path, 'bookends', 
                                            'field', str(z_offset)), 
                                       data_type='int16', num_channels=2,
                                       fill_missing=True, overwrite=True)
  pair_fields = {}
  for z_offset in pair_offsets.keys():
    for fr in forward_reverse:
      z_offset = z_offset * fr
      pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True)

  chunks = a.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                 cm.dst_voxel_offsets[mip], mip=mip, 
                                 max_mip=cm.num_scales)
  n_chunks = len(chunks)

  ###########################
  # Serial alignment script #
  ###########################
  
  # Copy first section
  batch = []
  for block_offset in copy_range:
    prefix = block_offset 
    for block_start in block_range:
      z = block_start + block_offset 
      print('copying z={0}'.format(z))
      t = a.copy(cm, src, dst, z, z, bbox, mip, is_field=False, prefix=prefix)
      batch.extend(t)

  run(a, batch)
  # wait
  start = time()
  for block_offset in copy_range:
    prefix = block_offset 
    for block_start in block_range:
      n = n_chunks
      a.wait_for_queue_empty(dst.path, 'copy_done/{}'.format(prefix), n)
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Serially align outward from the copied sections
  for block_offset in serial_range:
    batch = []
    for fr in forward_reverse:
      prefix = block_offset * fr
      z_offset = serial_offset * fr
      serial_field = serial_fields[z_offset]
      for block_start in block_range:
        z = block_start + block_offset * fr
        t = a.compute_field(cm, args.model_path, src, dst, serial_field, 
                            z, z+z_offset, bbox, mip, pad, prefix=prefix)
        batch.extend(t)

    run(a, batch)
    start = time()
    # wait 
    for fr in forward_reverse:
      prefix = block_offset * fr
      z_offset = serial_offset * fr
      serial_field = serial_fields[z_offset]
      n = n_chunks * len(block_range) * len(forward_reverse)
      a.wait_for_queue_empty(serial_field.path, 'compute_field_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    for fr in forward_reverse:
      prefix = block_offset * fr
      z_offset = serial_offset * fr
      serial_field = serial_fields[z_offset]
      for block_start in block_range:
        z = block_start + block_offset * fr
        t = a.render(cm, src, serial_field, dst, src_z=z, field_z=z, dst_z=z, 
                     bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
        batch.extend(t)

    run(a, batch)
    start = time()
    # wait 
    n = len(batch)
    a.wait_for_queue_empty(dst.path, 'render_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

  # Compute pair vector fields between the neighboring raw & serial sections
  for block_offset in pair_range:
    batch = []
    offset_range = pair_offsets[block_offset]
    for fr in forward_reverse:
      prefix = block_offset * fr
      for block_start in block_range:
        z = block_start + block_offset * fr
        for z_offset in offset_range:
          z_offset = z_offset * fr
          field = pair_fields[z_offset]
          t = a.compute_field(cm, args.model_path, src, dst, field, 
                              z+z_offset, z, bbox, mip, pad, prefix=prefix)
          batch.extend(t)

    run(a, batch)
    start = time()
    # wait 
    for fr in forward_reverse:
      prefix = block_offset * fr
      for z_offset in offset_range:
        z_offset = z_offset * fr
        field = pair_fields[z_offset]
        n = len(block_range) * n_chunks
        a.wait_for_queue_empty(field.path, 
            'compute_field_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

  # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)
