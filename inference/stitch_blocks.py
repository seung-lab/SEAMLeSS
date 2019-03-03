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
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--z_start', type=int)
  parser.add_argument('--z_stop', type=int)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--tgt_radius', type=int, default=3,
    help='int for number of sections to include in vector voting')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--odd_start', 
     help='indicate that the first block is odd (default is even)',
     action='store_true')
  args = parse_args(parser)
  # Only compute matches to previous sections
  a = get_aligner(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad
  src_mask_val = 1
  src_mask_mip = 8

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)
  
  # compile bbox & model lookup per z index
  bbox_lookup = {}
  model_lookup = {}
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
         for z in range(z_start, z_stop):
           bbox_lookup[z] = bbox 
           model_lookup[z] = model_path

  # Compile ranges
  block_range = range(args.z_start, args.z_stop, args.block_size)
  overlap = args.tgt_radius
  full_range = range(args.block_size + 2*overlap)

  copy_range = full_range[-overlap:]
  overlap_range = full_range[-2*overlap:-overlap][::-1]
  copy_field_range = range(overlap, args.block_size+overlap)
  broadcast_field_range = range(overlap-1, args.block_size+overlap)
  overlap_offsets = [i for i in range(1, overlap+1)]

  print('overlap_range {}'.format(overlap_range))
  print('overlap_offsets {}'.format(overlap_offsets))

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                  fill_missing=True, overwrite=False)
  src_mask_cv = None
  tgt_mask_cv = None

  # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius 
  dsts = {}
  block_types = ['even', 'odd']
  if args.odd_start:
    block_types = block_types[::-1]
  for block_type in block_types:
    dst = cm.create(join(args.dst_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=False)
    dsts[block_type] = dst 

  # Create field CloudVolumes
  pair_fields = {}
  for z_offset in overlap_offsets:
    pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', 
                                           'stitch_reverse', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True)
  temp_vvote_field = cm.create(join(args.dst_path, 'field', 'stitch_reverse', 'vvote', 'field'), 
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=True)
  temp_vvote_image = cm.create(join(args.dst_path, 'field', 'stitch_reverse', 'vvote', 'image'), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True)
  stitch_fields = {}
  for z_offset in overlap_offsets:
    stitch_fields[z_offset] = cm.create(join(args.dst_path, 'field', 
                                             'stitch_reverse', 'vvote', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True)
  broadcasting_field = cm.create(join(args.dst_path, 'field', 'stitch_reverse', 
                                      'broadcasting'),
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=True)
  block_field = cm.create(join(args.dst_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)

  compose_field = cm.create(join(args.dst_path, 'field', 'stitch_reverse', 'compose'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)
  final_dst = cm.create(join(args.dst_path, 'image_compose_reverse_test'), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True)

  ###################################################################
  # Create multiple fields aligning current block to next block #
  ###################################################################

  # Copy initial overlap sections (for this test) 
  batch = []
  for block_offset in copy_range:
    prefix = block_offset
    for i, block_start in enumerate(block_range):
      next_block_type = block_types[(i+1) % 2]
      next_block = dsts[next_block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.copy(cm, next_block, temp_vvote_image, z, z, bbox, mip, 
                     is_field=False, mask_cv=src_mask_cv, mask_mip=src_mask_mip, 
                     mask_val=src_mask_val, prefix=prefix)
      batch.extend(t)

  print('Scheduling CopyTasks')
  start = time()
  run(a, batch)
  end = time()
  diff = end - start
  print_run(diff, len(batch))
  # wait
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Vector vote the last sections of current block with first sections of next block
  for block_offset in overlap_range:
    print('BLOCK OFFSET {}'.format(block_offset))
    batch = []
    prefix = block_offset
    for i, block_start in enumerate(block_range):
      current_block_type = block_types[i % 2]
      next_block_type = block_types[(i+1) % 2]
      current_block = dsts[current_block_type]
      next_block = dsts[next_block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      model_path = model_lookup[z]
      for z_offset in overlap_offsets:
        field = pair_fields[z_offset]
        t = a.compute_field(cm, model_path, current_block, temp_vvote_image, field, 
                            z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
                            src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                            tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                            tgt_mask_val=src_mask_val, prefix=prefix,
                            prev_field_cv=block_field, prev_field_z=z,
                            prev_field_inverse=True)
        batch.extend(t)

    print('\nScheduling ComputeFieldTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait 
    print('block offset {}'.format(block_offset))
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    for block_start in block_range:
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.vector_vote(cm, pair_fields, temp_vvote_field, z, bbox, mip, inverse=False, 
                        serial=True, prefix=prefix)
      batch.extend(t)

    print('\nScheduling VectorVoteTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    # wait 
    print('block offset {}'.format(block_offset))
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print_run(diff, len(batch))

    batch = []
    task_counter = {}
    for i, block_start in enumerate(block_range):
      current_block_type = block_types[i % 2]
      current_block = dsts[current_block_type]
      z = block_start + block_offset 
      bbox = bbox_lookup[z]
      t = a.render(cm, current_block, temp_vvote_field, temp_vvote_image, 
                   src_z=z, field_z=z, dst_z=z, bbox=bbox, src_mip=mip, field_mip=mip, 
                   mask_cv=src_mask_cv, mask_val=src_mask_val, mask_mip=src_mask_mip,
                   prefix=prefix)
      batch.extend(t)

    print('\nScheduling RenderTasks')
    start = time()
    run(a, batch)
    end = time()
    diff = end - start
    print_run(diff, len(batch))
    start = time()
    print('block offset {}'.format(block_offset))
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print_run(diff, len(batch))

  #############################################################################
  # Combine multiple fields aligning current block to previous block into one #
  #############################################################################

  # Copy vvote fields to be at the same z index, so they can be vvoted again
  # TODO: Modify the ComputeFieldTask to make dst_z a parameter
  batch = []
  for i, block_start in enumerate(block_range):
    prefix = block_start
    for j, block_offset in enumerate(overlap_range):
      z = block_start + block_offset
      z_offset = j+1
      stitch_field = stitch_fields[z_offset]
      bbox = bbox_lookup[z]
      t = a.copy(cm, temp_vvote_field, stitch_field, z, block_start, bbox, mip, 
                     is_field=True, prefix=prefix)
      batch.extend(t)

  run(a, batch)
  start = time()
  # wait
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  # Vector vote these fields together to remove any folds
  batch = []
  for block_start in block_range:
    z = block_start
    bbox = bbox_lookup[z]
    t = a.vector_vote(cm, stitch_fields, broadcasting_field, z, bbox, mip, inverse=False, 
                      serial=True, prefix=prefix)
    batch.extend(t)

  print('\nScheduling VectorVoteTasks')
  start = time()
  run(a, batch)
  end = time()
  diff = end - start
  print_run(diff, len(batch))
  start = time()
  # wait 
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print_run(diff, len(batch))


  ###########################
  # Serial broadcast script #
  ###########################
  
  # # Copy vector field of first block
  # batch = []
  # block_start = block_range[0]
  # prefix = block_start
  # for block_offset in copy_field_range: 
  #   z = block_start + block_offset
  #   bbox = bbox_lookup[z]
  #   t = a.copy(cm, block_field, compose_field, z, z, bbox, mip, is_field=True, prefix=prefix)
  #   batch.extend(t)

  # run(a, batch)
  # start = time()
  # # wait
  # a.wait_for_sqs_empty()
  # end = time()
  # diff = end - start
  # print_run(diff, len(batch))

  # # Render out the images from the copied field
  # batch = []
  # block_start = block_range[0]
  # prefix = block_start
  # for block_offset in copy_field_range: 
  #   z = block_start + block_offset 
  #   bbox = bbox_lookup[z]
  #   t = a.render(cm, src, compose_field, final_dst, src_z=z, field_z=z, dst_z=z, 
  #                bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
  #   batch.extend(t)

  # print('Scheduling render for copied range')
  # start = time()
  # run(a, batch)
  # end = time()
  # diff = end - start
  # print_run(diff, len(batch))

  # Compose next block with last vector field from the previous composed block
  n_tasks = 0
  prefix = '' 
  start = time()
  # for i, block_start in enumerate(block_range[1:]):
  for i, block_start in enumerate(block_range):
    batch = []
    # z_broadcast = block_start + overlap - 1
    z_broadcast = block_start 
    for block_offset in broadcast_field_range:
      # decay the composition over the length of the block
      fixed_z = float(broadcast_field_range[0])
      factor = (block_offset - fixed_z) / (broadcast_field_range[-1] - fixed_z)
      z = block_start + block_offset
      bbox = bbox_lookup[z]
      t = a.compose(cm, broadcasting_field, block_field, compose_field, z_broadcast, z, z, 
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
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print_run(diff, len(batch))

  prefix = ''
  start = time()
  for i, block_start in enumerate(block_range):
    batch = []
    for block_offset in broadcast_field_range: 
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

