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
from itertools import compress
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
    range_list = []
    for i in range(part-1):
        range_list.append(block_range[i*srange:(i+1)*srange])
    range_list.append(block_range[(part-1)*srange:])
    return range_list

def ranges_overlap(a_pair, b_pair):
  a_start, a_stop = a_pair
  b_start, b_stop = b_pair
  return ((b_start <= a_start and b_stop >= a_start) or
         (b_start >= a_start and b_stop <= a_stop) or
         (b_start <= a_stop  and b_stop >= a_stop))


if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--model_lookup', type=str,
    help='relative path to CSV file identifying model to use per z range')
  parser.add_argument('--z_range_path', type=str, 
    help='path to csv file with list of z indices to use')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--z_start', type=int)
  parser.add_argument('--z_stop', type=int)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--overlap', type=int, default=3,
    help='int for number of sections in overlap between even & odd blocks')
  parser.add_argument('--stitching_vvote', type=int, default=3,
    help='int for number of sections to include in vector voting')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--decay_size', type=int, default=10)
  parser.add_argument('--min_interface', type=int, default=0,
    help='the minimum index of a block to be used for broadcasting')
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
  overlap = args.overlap
  stitching_vvote = args.stitching_vvote 
  full_range = range(args.block_size + overlap + stitching_vvote)

  complete_block_range = range(args.z_start, args.z_stop, args.block_size)
  complete_block_index = range(len(complete_block_range))
  working_block_range = complete_block_range
  working_block_index = complete_block_index
  even_odd_range = [i % 2 for i in range(len(working_block_range))]
  if args.z_range_path:
    print('Compiling z_range from {}'.format(args.z_range_path))
    decay_endpoints = range(args.z_start, args.z_stop+args.block_size+args.decay_size, 
                            args.block_size)
    block_pairs = list(zip(decay_endpoints[:-overlap], decay_endpoints[overlap:]))
    tmp_block_range = []
    tmp_block_index = []
    tmp_even_odd_range = []
    with open(args.z_range_path) as f:
      reader = csv.reader(f, delimiter=',')
      for k, r in enumerate(reader):
         if k != 0:
           z_pair = int(r[0]), int(r[1])
           print('Filtering block_range by {}'.format(z_pair))
           block_filter = [ranges_overlap(z_pair, b_pair) for b_pair in block_pairs]
           affected_blocks = list(compress(complete_block_range, block_filter))
           affected_blocks_index = list(compress(complete_block_index, block_filter))
           affected_even_odd = list(compress(even_odd_range, block_filter))
           print('Affected block_starts {}'.format(affected_blocks))
           tmp_block_range.extend(affected_blocks)
           tmp_block_index.extend(affected_blocks_index)
           tmp_even_odd_range.extend(affected_even_odd)
    working_block_range = tmp_block_range
    working_block_index = tmp_block_index
    even_odd_range = tmp_even_odd_range

  print('working_block_range {}'.format(working_block_range))
  print('working_block_index {}'.format(working_block_index))
  print('even_odd_range {}'.format(even_odd_range))

  copy_range = full_range[-overlap:]
  overlap_range = full_range[-overlap-stitching_vvote:-overlap][::-1]
  decay_start = overlap
  decay_stop = overlap + args.decay_size
  broadcast_field_range = range(overlap, args.block_size+overlap)
  overlap_vvote_offsets = [i for i in range(1, overlap+1)]
  broadcast_vvote_offsets = [i for i in range(1, stitching_vvote+1)]
  if args.forward_compose:
    copy_range = full_range[:overlap]
    overlap_range = full_range[overlap:overlap+stitching_vvote]
    broadcast_field_range = range(overlap, args.block_size+overlap)
    overlap_vvote_offsets = [-i for i in range(1, overlap+1)]
    broadcast_vvote_offsets = [-i for i in range(1, stitching_vvote+1)]

  print('complete_block_range {}'.format(complete_block_range))
  print('overlap_range {}'.format(overlap_range))
  print('overlap_vvote_offsets {}'.format(overlap_vvote_offsets))
  print('broadcast_vvote_offsets {}'.format(broadcast_vvote_offsets))
  print('broadcast_field_range {}'.format(broadcast_field_range))

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
  for i, block_type in enumerate(block_types):
    dst = cm.create(join(args.dst_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=False).path
    dsts[i] = dst 

  # Create field CloudVolumes
  pair_fields = {}
  for z_offset in overlap_vvote_offsets:
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
  for z_offset in broadcast_vvote_offsets:
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
      def __init__(self, brange, even_odd):
          self.brange = brange
          self.even_odd = even_odd
      def __iter__(self):
          for block_offset in copy_range:
            prefix = block_offset
            for block_start, even_odd in zip(self.brange, self.even_odd):
              prev_block = dsts[even_odd + 1 % 2]
              z = block_start + block_offset 
              bbox = bbox_lookup[z]
              t = a.copy(cm, prev_block, temp_vvote_image, z, z, bbox, mip, 
                             is_field=False, mask_cv=src_mask_cv, mask_mip=src_mask_mip, 
                             mask_val=src_mask_val, prefix=prefix) 
              yield from t

  print('Scheduling CopyTasks')
  range_list = make_range(working_block_range, a.threads)
  even_odd_list = make_range(even_odd_range, a.threads)

  ptask = []
  start = time()
  for irange, ieven_odd in zip(range_list, even_odd_list):
      ptask.append(CopyTaskIteratorImage(irange, ieven_odd))

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
      def __init__(self, brange, even_odd):
          self.brange = brange
          self.even_odd = even_odd
      def __iter__(self):
          for block_offset in copy_range:
            prefix = block_offset
            for block_start, even_odd in zip(self.brange, self.even_odd):
              z = block_start + block_offset 
              bbox = bbox_lookup[z]
              t = a.copy(cm, block_field, temp_vvote_field, z, z, bbox, mip, 
                             is_field=True, prefix=prefix) 
              yield from t

  print('Scheduling CopyTasks')
  ptask = []
  start = time()
  for irange, ieven_odd in zip(range_list, even_odd_list):
      ptask.append(CopyTaskIteratorField(irange, ieven_odd))

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
        def __init__(self, brange, even_odd):
            self.brange = brange
            self.even_odd = even_odd
        def __iter__(self):
            for block_start, even_odd in zip(self.brange, self.even_odd):
              current_block = dsts[even_odd]
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
              for z_offset in overlap_vvote_offsets:
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
    for irange, ieven_odd in zip(range_list, even_odd_list):
        ptask.append(ComputeFieldTaskIterator(irange, ieven_odd))

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
                                serial=True, prefix=prefix, softmin_temp=2**mip,
                                blur_sigma=1)
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
    print("Executing vvote use time:", diff)

    class RenderTaskIterator(object):
        def __init__(self, brange, even_odd):
            self.brange = brange
            self.even_odd = even_odd
        def __iter__(self):
            for block_start, even_odd in zip(self.brange, self.even_odd):
              current_block = dsts[even_odd]
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
    for irange, ieven_odd in zip(range_list, even_odd_list):
        ptask.append(RenderTaskIterator(irange, ieven_odd))

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
      def __init__(self, brange):
          self.brange = brange
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
  for irange in range_list:
      ptask.append(CopyTaskIteratorII(irange))

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
                            serial=True, prefix=prefix, softmin_temp=2**mip,
                            blur_sigma=1)
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
  print("Executing vvote use time:", diff)

  ###########################
  # Serial broadcast script #
  ###########################
  
  # Compose next block with last vector field from the previous composed block

  def interpolate(x, start, stop):
    """Return interpolation value of x for range(start, stop)

    Args
       x: int location
       start: location corresponding to 1
       stop: location corresponding to 0 
    """
    assert(stop != start)
    d = (stop - x) / (stop - start) 
    return min(max(d, 0.), 1.)

  broadcast_range_list = make_range(broadcast_field_range, a.threads)
  prefix = '' 
  for j, block_start in zip(working_block_index, working_block_range):
    # z_broadcast = block_start
    i = max(j - int(math.ceil(args.decay_size / args.block_size)) + 1, args.min_interface)
    block_starts = complete_block_range[i:j+1]
    bcast_tuples = [(x+decay_start, x+decay_stop) for x in block_starts] 
    class ComposeTaskIterator(object):
      def __init__(self, brange):
        self.brange = brange
      def __iter__(self):
        for block_offset in self.brange:
          z = block_start + block_offset
          print('z {}'.format(z))
          factors = [interpolate(z, bstart, bstop) for (bstart, bstop) in bcast_tuples]
          factors += [1.]
          print('factors {}'.format(factors))
          bbox = bbox_lookup[z]
          cv_list = [broadcasting_field]*len(block_starts) + [block_field]
          z_list = list(block_starts) + [z]
          t = a.multi_compose(cm, cv_list, compose_field, z_list, z, bbox, 
                              mip, mip, factors, pad, prefix=prefix)
          yield from t

    ptask = []
    start = time()
    for irange in broadcast_range_list:
        print('irange {}'.format(irange))
        ptask.append(ComposeTaskIterator(irange))

    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
    print('Scheduling compose for block_start {}, block {} / {}'.format(block_start, i+1, 
                                                                len(working_block_range)))
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
  for i, block_start in enumerate(working_block_range):
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
                                                               len(working_block_range)))
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

