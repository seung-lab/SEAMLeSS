"""
Stitch even and odd blocks together
T Macrina
190305

TODO: This script does not properly handle the last block (reverse compose)
      or the first block (forward compose). Proper handling should be put
      in place.
"""
import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue

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

def make_range(block_range, part_num):
    rangelen = len(block_range)
    if(rangelen < part_num):
        srange =1
        part = rangelen
    else:
        part = part_num
        srange = rangelen//part
    if srange%2 == 0:
        odd_even = 0
    else:
        odd_even = 1
    range_list = []
    for i in range(part-1):
        range_list.append(block_range[i*srange:(i+1)*srange])
    range_list.append(block_range[(part-1)*srange:])
    return range_list, odd_even

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
  parser.add_argument('--forward_compose', 
     help='stitch blocks by broadcast composing vector field from previous block across' \
          ' current block (default is to reverse compose from next block across current)',
     action='store_true')
  parser.add_argument('--suffix', type=str, default='',
     help='string to append to directory names')
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
  broadcast_field_range = range(overlap, args.block_size+overlap)
  overlap_offsets = [i for i in range(1, overlap+1)]
  if args.forward_compose:
    copy_range = full_range[:overlap]
    overlap_range = full_range[overlap:2*overlap]
    copy_field_range = range(overlap, args.block_size+overlap)
    broadcast_field_range = range(overlap, args.block_size+overlap)
    overlap_offsets = [-i for i in range(1, overlap+1)]

  print('overlap_range {}'.format(overlap_range))
  print('overlap_offsets {}'.format(overlap_offsets))

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                  fill_missing=True, overwrite=False).path
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
                    overwrite=False).path
    dsts[block_type] = dst 

  # Create field CloudVolumes
  pair_fields = {}
  for z_offset in overlap_offsets:
    pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', 
                                           'stitch{}'.format(args.suffix), str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True).path
  temp_vvote_field = cm.create(join(args.dst_path, 'field', 'stitch{}'.format(args.suffix), 
                                    'vvote', 'field'), 
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=True).path
  temp_vvote_image = cm.create(join(args.dst_path, 'field', 'stitch{}'.format(args.suffix), 
                                    'vvote', 'image'), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True).path
  stitch_fields = {}
  for z_offset in overlap_offsets:
    stitch_fields[z_offset] = cm.create(join(args.dst_path, 'field', 
                                             'stitch{}'.format(args.suffix), 
                                             'vvote', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True).path
  broadcasting_field = cm.create(join(args.dst_path, 'field', 
                                      'stitch{}'.format(args.suffix), 
                                      'broadcasting'),
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=True).path
  block_field = cm.create(join(args.dst_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False).path

  compose_field = cm.create(join(args.dst_path, 'field', 'stitch{}'.format(args.suffix), 
                                 'compose'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True).path
  final_dst = cm.create(join(args.dst_path, 'image_stitch{}'.format(args.suffix)), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True).path

  ###################################################################
  # Create multiple fields aligning current block to next block #
  ###################################################################

  # Copy initial overlap sections (for this test) 

  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

  class CopyTaskIteratorImage():
      def __init__(self, brange, odd_even):
          self.brange = brange
          self.odd_even = odd_even
      def __iter__(self):
          for block_offset in copy_range:
            prefix = block_offset
            for i, block_start in enumerate(self.brange):
              next_block_type = block_types[(i+self.odd_even+1) % 2]
              next_block = dsts[next_block_type]
              z = block_start + block_offset 
              bbox = bbox_lookup[z]
              t = a.copy(cm, next_block, temp_vvote_image, z, z, bbox, mip, 
                             is_field=False, mask_cv=src_mask_cv, mask_mip=src_mask_mip, 
                             mask_val=src_mask_val, prefix=prefix) 
              yield from t

  print('Scheduling CopyTasks')
  range_list, odd_even = make_range(block_range, a.threads)

  ptask = []
  start = time()
  for i, irange in enumerate(range_list):
      ptask.append(CopyTaskIteratorImage(irange, i*odd_even))

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)
  end = time()
  diff = end - start
  print("Sending Copy Tasks use time:", diff)
  # wait
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Copy Tasks use time:", diff) 

  class CopyTaskIteratorField():
      def __init__(self, brange, odd_even):
          self.brange = brange
          self.odd_even = odd_even
      def __iter__(self):
          for block_offset in copy_range:
            prefix = block_offset
            for i, block_start in enumerate(self.brange):
              next_block_type = block_types[(i+self.odd_even+1) % 2]
              next_block = dsts[next_block_type]
              z = block_start + block_offset 
              bbox = bbox_lookup[z]
              t = a.copy(cm, block_field, temp_vvote_field, z, z, bbox, mip, 
                             is_field=True, prefix=prefix) 
              yield from t

  print('Scheduling CopyTasks')
  range_list, odd_even = make_range(block_range, a.threads)

  ptask = []
  start = time()
  for i, irange in enumerate(range_list):
      ptask.append(CopyTaskIteratorField(irange, i*odd_even))

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)
  end = time()
  diff = end - start
  print("Sending Copy Tasks use time:", diff)
  # wait
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Copy Tasks use time:", diff) 

  # Vector vote the last sections of current block with first sections of next block
  for block_offset in overlap_range:
    print('BLOCK OFFSET {}'.format(block_offset))
    prefix = block_offset

    class ComputeFieldTaskIterator(object):
        def __init__(self, brange, odd_even):
            self.brange = brange
            self.odd_even = odd_even
        def __iter__(self):
            for i, block_start in enumerate(self.brange): 
              current_block_type = block_types[(i+self.odd_even)% 2]
              next_block_type = block_types[(i+self.odd_even+1) % 2]
              current_block = dsts[current_block_type]
              next_block = dsts[next_block_type]
              z = block_start + block_offset 
              prev_field_z = z
              prev_field_cv = temp_vvote_field
              prev_field_inverse = False
              if not args.forward_compose:
                prev_field_cv = block_field
                prev_field_z = z
                prev_field_inverse = True 
                if block_offset != overlap_range[0]:
                  prev_field_cv = temp_vvote_field
                  prev_field_z = z + 1
                  prev_field_inverse = False
              bbox = bbox_lookup[z]
              model_path = model_lookup[z]
              for z_offset in overlap_offsets:
                field = pair_fields[z_offset]
                if args.forward_compose:
                  prev_field_z = z+z_offset
                t = a.compute_field(cm, model_path, current_block, temp_vvote_image, field, 
                                    z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
                                    src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                                    tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                                    tgt_mask_val=src_mask_val, prefix=prefix,
                                    prev_field_cv=prev_field_cv, prev_field_z=prev_field_z,
                                    prev_field_inverse=prev_field_inverse)
                yield from t

    print('\nScheduling ComputeFieldTasks')
    start = time() 
    ptask = []
    for i, irange in enumerate(range_list):
        ptask.append(ComputeFieldTaskIterator(irange, i*odd_even))

    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
    end = time()
    diff = end - start
    print("Sending Compute Field Tasks use time:", diff)
    print('Running Compute field')
    start = time()
    # wait 
    print('block offset {}'.format(block_offset))
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print("Executing Compute Tasks use time:", diff)

    class VvoteTaskIterator(object):
        def __init__(self, brange):
            self.brange = brange
        def __iter__(self):
            for block_start in self.brange:
              z = block_start + block_offset
              bbox = bbox_lookup[z]
              t = a.vector_vote(cm, pair_fields, temp_vvote_field,
                                z, bbox, mip, inverse=False,
                                serial=True, prefix=prefix)
              yield from t
    print('\nScheduling VectorVoteTasks')
    ptask = []
    start = time()
    for irange in range_list:
        ptask.append(VvoteTaskIterator(irange))

    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
    end = time()
    diff = end - start
    print("Sending Vvote Tasks for vvote use time:", diff)
    print('Run vvoting')
    start = time()
    # wait 
    print('block offset {}'.format(block_offset))
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print("Executing vvtote use time:", diff)

    class RenderTaskIterator(object):
        def __init__(self, brange, odd_even):
            self.brange = brange
            self.odd_even = odd_even
        def __iter__(self):
            for i, block_start in enumerate(self.brange):
              current_block_type = block_types[(i+self.odd_even) % 2]
              current_block = dsts[current_block_type]
              z = block_start + block_offset
              bbox = bbox_lookup[z]
              t = a.render(cm, current_block, temp_vvote_field, temp_vvote_image, 
                           src_z=z, field_z=z, dst_z=z, bbox=bbox, src_mip=mip, field_mip=mip, 
                           mask_cv=src_mask_cv, mask_val=src_mask_val, mask_mip=src_mask_mip,
                           prefix=prefix)
              yield from t
    print('\nScheduling RenderTasks')
    ptask = []
    start = time()
    for i, irange in enumerate(range_list):
        ptask.append(RenderTaskIterator(irange, i*odd_even))

    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)

    end = time()
    diff = end - start
    print("Sending Render Tasks use time:", diff)
    print('Run rendering')
    start = time()
    print('block offset {}'.format(block_offset))
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print("Executing Rendering use time:", diff)

  #############################################################################
  # Combine multiple fields aligning current block to previous block into one #
  #############################################################################

  # Copy vvote fields to be at the same z index, so they can be vvoted again
  # TODO: Modify the ComputeFieldTask to make dst_z a parameter

  class CopyTaskIteratorII():
      def __init__(self, brange, odd_even):
          self.brange = brange
          self.odd_even = odd_even
      def __iter__(self):
          for i, block_start in enumerate(self.brange):
            prefix = block_start
            for j, block_offset in enumerate(overlap_range):
              z = block_start + block_offset
              z_offset = j+1
              if args.forward_compose:
                z_offset = -1*z_offset
              stitch_field = stitch_fields[z_offset]
              bbox = bbox_lookup[z]
              t = a.copy(cm, temp_vvote_field, stitch_field, z, block_start, bbox, mip, 
                             is_field=True, prefix=prefix)
              yield from t
  ptask = []
  start = time() 
  for i, irange in enumerate(range_list):
      ptask.append(CopyTaskIteratorII(irange, i*odd_even))

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)

  end = time()
  diff = end - start
  print("Sending Copy Tasks use time:", diff)
  start = time()
  # wait
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Copy task use time:", diff)

  # Vector vote these fields together to remove any folds
  class VvoteTaskIteratorII(object):
    def __init__(self, brange):
        self.brange = brange
    def __iter__(self):
        for block_start in self.brange:
          z = block_start
          bbox = bbox_lookup[z]
          t = a.vector_vote(cm, stitch_fields, broadcasting_field,
                            z, bbox, mip, inverse=False,
                            serial=True, prefix=prefix)
          yield from t

  print('\nScheduling VectorVoteTasks')
  ptask = []
  start = time()
  for irange in range_list:
      ptask.append(VvoteTaskIteratorII(irange))

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)
  end = time()
  diff = end - start
  print("Sending Vvote Tasks for vvote use time:", diff)
  print('Run vvoting')
 
  start = time()
  # wait 
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing vvtote use time:", diff)

  ###########################
  # Serial broadcast script #
  ###########################
  
  # Compose next block with last vector field from the previous composed block
  prefix = '' 
  # for i, block_start in enumerate(block_range[1:]):
  
  for i, block_start in enumerate(block_range):
    # z_broadcast = block_start + overlap - 1
    z_broadcast = block_start
    class ComposeTaskIterator(object):
        def __init__(self, brange):
            self.brange = brange
        def __iter__(self):
            for block_offset in self.brange:
              fixed_z = float(broadcast_field_range[0])
              factor = (block_offset - fixed_z) / (broadcast_field_range[-1] - fixed_z)
              if args.forward_compose:
                last_z = float(broadcast_field_range[-1])
                factor = (last_z - block_offset) / (last_z - broadcast_field_range[0])
              z = block_start + block_offset
              bbox = bbox_lookup[z]
              t = a.compose(cm, broadcasting_field, block_field, compose_field,
                            z_broadcast, z, z, bbox, mip, mip, mip, factor,
                            prefix=prefix)
              yield from t

    broadcast_range_list, brodd_even = make_range(broadcast_field_range[:8], a.threads)
    ptask = []
    start = time()
    for irange in broadcast_range_list:
        ptask.append(ComposeTaskIterator(irange))

    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
    print('Scheduling compose for block_start {}, block {} / {}'.format(block_start, i+1, 
                                                                    len(block_range[1:])))
    end = time()
    diff = end - start
    print("Sending Compose Tasks use time:", diff)
  # wait
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Compose tasks use time:", diff)

  prefix = ''
  start = time()
  for i, block_start in enumerate(block_range):
    class RenderTaskIteratorII(object):
        def __init__(self, brange):
            self.brange = brange
        def __iter__(self): 
            for block_offset in self.brange:
              z = block_start + block_offset 
              bbox = bbox_lookup[z]
              t = a.render(cm, src, compose_field, final_dst, src_z=z, field_z=z, dst_z=z, 
                           bbox=bbox, src_mip=mip, field_mip=mip, prefix=prefix)
              yield from t
    print('Scheduling render for block_start {}, block {} / {}'.format(block_start, i+1, 
                                                                    len(block_range[1:])))
    start = time()
    ptask = []
    for irange in broadcast_range_list:
        ptask.append(RenderTaskIteratorII(irange))
  
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
    end = time()
    diff = end - start
    print("Sending Render Tasks use time:", diff)
 
  start = time()
  # wait 
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Rendering for copied range use time:", diff)

