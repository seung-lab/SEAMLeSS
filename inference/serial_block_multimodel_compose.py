import sys
import torch
import json
import math
import csv
from time import time, sleep
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from tasks import run 
from boundingbox import BoundingBox

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--model_lookup', type=str,
    help='relative path to CSV file identifying model to use per z range')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--tgt_radius', type=int, default=3,
    help='int for number of sections to include in vector voting')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--restart', type=int, default=0)
  parser.add_argument('--use_sqs_wait', action='store_true',
    help='wait for SQS to return that its queue is empty; incurs fixed 30s for initial wait')
  args = parse_args(parser)
  # Only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad
  src_mask_val = args.src_mask_val
  src_mask_mip = args.src_mask_mip

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)
  
  # compile bbox & model lookup per z index
  bbox_lookup = {}
  model_lookup = {}
  z_min = math.inf
  z_max = -math.inf
  with open(args.model_lookup) as f:
    reader = csv.reader(f, delimiter=',')
    for k, r in enumerate(reader):
       if k != 0:
         x_start = int(r[0])
         y_start = int(r[1])
         z_start = int(r[2])
         x_stop  = int(r[3])
         y_stop  = int(r[4])
         z_stop  = int(r[5])
         bbox_mip = int(r[6])
         model_path = join('..', 'models', r[7])
         bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
         if z_start < z_min:
           z_min = z_start
         if z_stop > z_max:
           z_max= z_stop
         for z in range(z_start, z_stop):
           bbox_lookup[z] = bbox 
           model_lookup[z] = model_path

  # Compile ranges
  block_range = range(z_min, z_max, args.block_size)
  overlap = args.tgt_radius
  full_range = range(args.block_size + overlap)

  copy_range = full_range[overlap-1:overlap]
  serial_range = full_range[:overlap-1][::-1]
  vvote_range = full_range[overlap:]
  copy_field_range = range(overlap, args.block_size+overlap)
  broadcast_field_range = range(overlap-1, args.block_size+overlap)

  serial_offsets = {serial_range[i]: i+1 for i in range(overlap-1)}
  vvote_offsets = [-i for i in range(1, overlap+1)]

  print('copy_range {}'.format(copy_range))
  print('serial_range {}'.format(serial_range))
  print('vvote_range {}'.format(vvote_range))
  print('serial_offsets {}'.format(serial_offsets))
  print('vvote_offsets {}'.format(vvote_offsets))

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  src_mask_cv = None
  tgt_mask_cv = None
  if args.src_mask_path:
    src_mask_cv = cm.create(args.src_mask_path, data_type='uint8', num_channels=1,
                               fill_missing=True, overwrite=False)
    tgt_mask_cv = src_mask_cv

  # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius 
  dsts = {}
  block_types = ['even', 'odd']
  for block_type in block_types:
    dst = cm.create(join(args.dst_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True)
    dsts[block_type] = dst 

  # Create field CloudVolumes
  serial_fields = {}
  for z_offset in serial_offsets.values():
    serial_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                  data_type='int16', num_channels=2,
                                  fill_missing=True, overwrite=True)
  pair_fields = {}
  for z_offset in vvote_offsets:
    pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True)
  vvote_field = cm.create(join(args.dst_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  compose_field = cm.create(join(args.dst_path, 'field', 'block_compose'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)
  final_dst = cm.create(join(args.dst_path, 'image'), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True)

  ###########################
  # Serial alignment script #
  ###########################
  # check for restart
  copy_range = [r for r in copy_range if r >= args.restart]
  serial_range = [r for r in serial_range if r >= args.restart]
  vvote_range = [r for r in vvote_range if r >= args.restart]
  
  # Copy first section
  batch = []
  task_counter = {}
  for block_offset in copy_range:
    prefix = block_offset
    for i, block_start in enumerate(block_range):
      block_type = block_types[i % 2]
      dst = dsts[block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.copy(cm, src, dst, z, z, bbox, mip, is_field=False, mask_cv=src_mask_cv,
                     mask_mip=src_mask_mip, mask_val=src_mask_val, prefix=prefix)
      k = (prefix, dst.path)
      if k not in task_counter:
        task_counter[k] = 0
      task_counter[k] += len(t)
      batch.extend(t)
  print('Scheduling CopyTasks')
  start = time()
  run(a, batch)
  end = time()
  diff = end - start
  print_run(diff, len(batch))
  # wait
  start = time()
  if args.use_sqs_wait:
    a.wait_for_sqs_empty()
  else:
    for (prefix, path), n in task_counter.items():
      a.wait_for_queue_empty(path, 'copy_done/{}'.format(prefix), n)
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Align without vector voting
  for block_offset in serial_range:
    z_offset = serial_offsets[block_offset] 
    serial_field = serial_fields[z_offset]
    batch = []
    prefix = block_offset
    for i, block_start in enumerate(block_range):
      block_type = block_types[i % 2]
      dst = dsts[block_type]
      z = block_start + block_offset 
      model_path = model_lookup[z]
      bbox = bbox_lookup[z]
      t = a.compute_field(cm, model_path, src, dst, serial_field, 
                          z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
                          src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                          tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                          tgt_mask_val=src_mask_val, prefix=prefix)
      batch.extend(t)

    print('Scheduling ComputeFieldTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait 
    if args.use_sqs_wait:
      a.wait_for_sqs_empty()
    else:
      n = len(batch) 
      a.wait_for_queue_empty(serial_field.path, 
          'compute_field_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    task_counter = {}
    for i, block_start in enumerate(block_range):
      block_type = block_types[i % 2]
      dst = dsts[block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.render(cm, src, serial_field, dst, src_z=z, field_z=z, dst_z=z, 
                   bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
                   mask_val=src_mask_val, mask_mip=src_mask_mip, prefix=prefix)
      k = (prefix, dst.path)
      if k not in task_counter:
        task_counter[k] = 0
      task_counter[k] += len(t)
      batch.extend(t)

    print('Scheduling RenderTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait 
    if args.use_sqs_wait:
      a.wait_for_sqs_empty()
    else:
      for (prefix, path), n in task_counter.items():
        a.wait_for_queue_empty(path, 'render_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

  # Align with vector voting
  for block_offset in vvote_range:
    batch = []
    task_counter = {}
    prefix = block_offset
    for i, block_start in enumerate(block_range):
      block_type = block_types[i % 2]
      dst = dsts[block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      model_path = model_lookup[z]
      for z_offset in vvote_offsets:
        field = pair_fields[z_offset]
        t = a.compute_field(cm, model_path, src, dst, field, 
                            z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
                            src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                            tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                            tgt_mask_val=src_mask_val, prefix=prefix)
        batch.extend(t)
        k = (prefix, field.path)
        if k not in task_counter:
          task_counter[k] = 0
        task_counter[k] += len(t)

    print('Scheduling ComputeFieldTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait 
    if args.use_sqs_wait:
      a.wait_for_sqs_empty()
    else:
      for (prefix, path), n in task_counter.items():
        a.wait_for_queue_empty(path, 'compute_field_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    for block_start in block_range:
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.vector_vote(cm, pair_fields, vvote_field, z, bbox, mip, inverse=False, 
                        serial=True, prefix=prefix)
      batch.extend(t)

    print('Scheduling VectorVoteTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait 
    if args.use_sqs_wait:
      a.wait_for_sqs_empty()
    else:
      n = len(batch)
      a.wait_for_queue_empty(vvote_field.path, 
          'vector_vote_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    task_counter = {}
    for i, block_start in enumerate(block_range):
      block_type = block_types[i % 2]
      dst = dsts[block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.render(cm, src, vvote_field, dst, 
                   src_z=z, field_z=z, dst_z=z, bbox=bbox, src_mip=mip, field_mip=mip, 
                   mask_cv=src_mask_cv, mask_val=src_mask_val, mask_mip=src_mask_mip,
                   prefix=prefix)
      batch.extend(t)
      k = (prefix, dst.path)
      if k not in task_counter:
        task_counter[k] = 0
      task_counter[k] += len(t)

    print('Scheduling RenderTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait
    if args.use_sqs_wait:
      a.wait_for_sqs_empty()
    else:
      for (prefix, path), n in task_counter.items():
        a.wait_for_queue_empty(path, 'render_done/{}'.format(prefix), n)
    end = time()
    diff = end - start
    print_run(diff, len(batch))

  ###########################
  # Serial broadcast script #
  ###########################
  
  # Copy vector field of first block
  batch = []
  block_start = block_range[0]
  prefix = block_start
  for block_offset in copy_field_range: 
    z = block_start + block_offset
    bbox = bbox_lookup[z]
    t = a.copy(cm, vvote_field, compose_field, z, z, bbox, mip, is_field=True, prefix=prefix)
    batch.extend(t)

  run(a, batch)
  start = time()
  # wait
  if args.use_sqs_wait:
    a.wait_for_sqs_empty()
  else:
    a.wait_for_queue_empty(compose_field.path, 'copy_done/{}'.format(prefix), len(batch))
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Render out the images from the copied field
  batch = []
  block_start = block_range[0]
  prefix = block_start
  for block_offset in copy_field_range: 
    z = block_start + block_offset 
    bbox = bbox_lookup[z]
    t = a.render(cm, src, compose_field, final_dst, src_z=z, field_z=z, dst_z=z, 
                 bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
    batch.extend(t)

  print('Scheduling render for copied range')
  start = time()
  run(a, batch)
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Compose next block with last vector field from the previous composed block
  n_tasks = 0
  prefix = '' 
  start = time()
  for i, block_start in enumerate(block_range[1:]):
    batch = []
    z_broadcast = block_start + overlap - 1
    for block_offset in broadcast_field_range[1:]:
      # decay the composition over the length of the block
      br = float(broadcast_field_range[-1])
      factor = (br - block_offset) / (br - broadcast_field_range[1])
      z = block_start + block_offset
      bbox = bbox_lookup[z]
      t = a.compose(cm, vvote_field, vvote_field, compose_field, z_broadcast, z, z, 
                    bbox, mip, mip, mip, factor, prefix=prefix)
      batch.extend(t)
      n_tasks += len(t)

    print('Scheduling compose for block_start {}, block {} / {}'.format(block_start, i+1, 
                                                                    len(block_range[1:])))
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
  # wait
  if args.use_sqs_wait:
    a.wait_for_sqs_empty()
  else:
    a.wait_for_queue_empty(compose_field.path, 'compose_done/{}'.format(prefix), n_tasks)
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  prefix = ''
  start = time()
  for i, block_start in enumerate(block_range[1:]):
    batch = []
    for block_offset in broadcast_field_range[1:]: 
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.render(cm, src, compose_field, final_dst, src_z=z, field_z=z, dst_z=z, 
                   bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
      batch.extend(t)

    print('Scheduling render for block_start {}, block {} / {}'.format(block_start, i+1, 
                                                                    len(block_range[1:])))
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))


  # # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)
