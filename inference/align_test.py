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
from as_args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
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
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
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
  even_odd_range = [i % 2 for i in range(len(block_range))]
  if args.z_range_path:
    print('Compiling z_range from {}'.format(args.z_range_path))
    block_endpoints = range(args.z_start, args.z_stop+args.block_size, args.block_size)
    block_pairs = list(zip(block_endpoints[:-1], block_endpoints[1:]))
    tmp_block_range = []
    tmp_even_odd_range = []
    with open(args.z_range_path) as f:
      reader = csv.reader(f, delimiter=',')
      for k, r in enumerate(reader):
         if k != 0:
           z_pair = int(r[0]), int(r[1])
           print('Filtering block_range by {}'.format(z_pair))
           block_filter = [ranges_overlap(z_pair, b_pair) for b_pair in block_pairs]
           affected_blocks = list(compress(block_range, block_filter))
           affected_even_odd = list(compress(even_odd_range, block_filter))
           print('Affected block_starts {}'.format(affected_blocks))
           tmp_block_range.extend(affected_blocks)
           tmp_even_odd_range.extend(affected_even_odd)
    block_range = tmp_block_range
    even_odd_range = tmp_even_odd_range

  print('block_range {}'.format(block_range))
  print('even_odd_range {}'.format(even_odd_range))

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

  if src_mask_cv != None:
      src_mask_cv = src_mask_cv.path
  if tgt_mask_cv != None:
      tgt_mask_cv = tgt_mask_cv.path

  # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius 
  dsts = {}
  block_types = ['even', 'odd']
  for i, block_type in enumerate(block_types):
    dst = cm.create(join(args.dst_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=True)
    dsts[i] = dst.path 

  # Create field CloudVolumes
  serial_fields = {}
  for z_offset in serial_offsets.values():
    serial_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                  data_type='int16', num_channels=2,
                                  fill_missing=True, overwrite=True).path
  pair_fields = {}
  for z_offset in vvote_offsets:
    pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True).path
  vvote_field = cm.create(join(args.dst_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  ###########################
  # Serial alignment script #
  ###########################
  # check for restart
  copy_range = [r for r in copy_range if r >= args.restart]
  serial_range = [r for r in serial_range if r >= args.restart]
  vvote_range = [r for r in vvote_range if r >= args.restart]
  
  # Copy first section
  chunks = a.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, 
                                    max_mip=cm.max_mip)
  for c in chunks:
      print("------------------------> bbox is", c.stringify(0), mip)
  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)  
 
  class CopyTaskIterator():
      def __init__(self, brange, even_odd):
          self.brange = brange
          self.even_odd = even_odd
      def __iter__(self):
          print(self.brange)
          for block_offset in copy_range:
            prefix = block_offset
            for block_start, even_odd in zip(self.brange, self.even_odd):
              dst = dsts[even_odd]
              z = block_start + block_offset
              bbox = bbox_lookup[z]
              t =  a.copy(cm, src.path, dst, z, z, bbox, mip, is_field=False,
                         mask_cv=src_mask_cv, mask_mip=src_mask_mip, mask_val=src_mask_val,
                         prefix=prefix)
              yield from t 

  ptask = []
  range_list = make_range(block_range, a.threads)
  even_odd_list = make_range(even_odd_range, a.threads)
#  print('range_list {}'.format(range_list))
#  print('even_odd_list {}'.format(even_odd_list))
#  
#  start = time()
#  for irange, ieven_odd in zip(range_list, even_odd_list):
#      ptask.append(CopyTaskIterator(irange, ieven_odd))
#
#  with ProcessPoolExecutor(max_workers=a.threads) as executor:
#      executor.map(remote_upload, ptask)
# 
#  end = time()
#  diff = end - start
#  print("Sending Copy Tasks use time:", diff)
#  print('Run CopyTasks')
#  # wait
#  start = time()
#  #if args.use_sqs_wait:
#  a.wait_for_sqs_empty()
#  end = time()
#  diff = end - start
#  print("Executing Copy Tasks use time:", diff)
#

  # Align without vector voting
  for block_offset in serial_range:
    print('BLOCK OFFSET {}'.format(block_offset))
    z_offset = serial_offsets[block_offset] 
    serial_field = serial_fields[z_offset]
    prefix = block_offset
    class ComputeFieldTaskIterator(object):
        def __init__(self, brange, even_odd):
            self.brange = brange
            self.even_odd = even_odd
        def __iter__(self):
            for block_start, even_odd in zip(self.brange, self.even_odd):
                dst = dsts[even_odd]
                z = block_start + block_offset 
                model_path = model_lookup[z]
                bbox = bbox_lookup[z]
                t = a.compute_field(cm, model_path, src.path, dst, serial_field, 
                #t = a.compute_field(cm, model_path, src.path, src.path, serial_field, 
                                    z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
                                    src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                                    tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                                    tgt_mask_val=src_mask_val, prefix=prefix,
                                    prev_field_cv=None, prev_field_z=None)
                yield from t
#    ptask = []
#    start = time()
#    print("block_range", block_range)
#    for irange, ieven_odd in zip(range_list, even_odd_list):
#        ptask.append(ComputeFieldTaskIterator(irange, ieven_odd))
#    print("-----ptask len is", len(ptask), a.threads) 
#    with ProcessPoolExecutor(max_workers=a.threads) as executor:
#        executor.map(remote_upload, ptask)
#   
#    end = time()
#    diff = end - start
#    print("Sending Compute Field Tasks use time:", diff)
#    print('Run Compute field')
# 
#    start = time()
#    # wait 
#    a.wait_for_sqs_empty()
#    end = time()
#    diff = end - start
#    print("Executing Compute Tasks use time:", diff)

    class RenderTaskIterator(object):
        def __init__(self, brange, even_odd):
            self.brange = brange
            self.even_odd = even_odd
        def __iter__(self):
            for block_start, even_odd in zip(self.brange, self.even_odd):
                dst = dsts[even_odd]
                z = block_start + block_offset
                bbox = bbox_lookup[z]
                t = a.render(cm, src.path, serial_field, dst, src_z=z, field_z=z, dst_z=z,
                             bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
                             mask_val=src_mask_val, mask_mip=src_mask_mip, prefix=prefix)
                yield from t
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
    # wait 
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print("Executing Rendering use time:", diff)

  # Align with vector voting
#  for block_offset in vvote_range:
#    print('BLOCK OFFSET {}'.format(block_offset))
#    prefix = block_offset
#    class ComputeFieldTaskIteratorII(object):
#        def __init__(self, brange, even_odd):
#            self.brange = brange
#            self.even_odd = even_odd
#        def __iter__(self):
#            for block_start, even_odd in zip(self.brange, self.even_odd):
#                dst = dsts[even_odd]
#                z = block_start + block_offset 
#                bbox = bbox_lookup[z]
#                model_path = model_lookup[z]
#                for z_offset in vvote_offsets:
#                    field = pair_fields[z_offset]
#                    t = a.compute_field(cm, model_path, src.path, dst, field, 
#                                        z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
#                                        src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
#                                        tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
#                                        tgt_mask_val=src_mask_val, prefix=prefix, 
#                                        prev_field_cv=vvote_field.path, prev_field_z=z+z_offset)
#                    yield from t
#    ptask = []
#    start = time()
#    for irange, ieven_odd in zip(range_list, even_odd_list):
#        ptask.append(ComputeFieldTaskIteratorII(irange, ieven_odd))
#    
#    with ProcessPoolExecutor(max_workers=a.threads) as executor:
#        executor.map(remote_upload, ptask)
#   
#    end = time()
#    diff = end - start
#    print("Sending Compute Field Tasks for vvote use time:", diff)
#    print('Run Compute field for vvote')
#    # wait 
#    print('block offset {}'.format(block_offset))
#    start = time()
#    a.wait_for_sqs_empty()
#    end = time()
#    diff = end - start
#    print("Executing Compute Tasks for vvtote use time:", diff)
#
#    class VvoteTaskIterator(object):
#        def __init__(self, brange):
#            self.brange = brange
#        def __iter__(self):
#            for block_start in self.brange:
#                z = block_start + block_offset
#                bbox = bbox_lookup[z]
#                t = a.vector_vote(cm, pair_fields, vvote_field.path, z, bbox,
#                                  mip, inverse=False, serial=True, prefix=prefix)
#                yield from t
#    ptask = []
#    start = time()
#    for irange in range_list:
#        ptask.append(VvoteTaskIterator(irange))
#    
#    with ProcessPoolExecutor(max_workers=a.threads) as executor:
#        executor.map(remote_upload, ptask)
#   
#    end = time()
#    diff = end - start
#    print("Sending Vvote Tasks for vvote use time:", diff)
#    print('Run vvoting')
#    # wait 
#    print('block offset {}'.format(block_offset))
#    start = time()
#    a.wait_for_sqs_empty()
#    end = time()
#    diff = end - start
#    print("Executing vvtote use time:", diff)
#
#    class RenderTaskIteratorII(object):
#        def __init__(self, brange, even_odd):
#            self.brange = brange
#            self.even_odd = even_odd
#        def __iter__(self):
#            for block_start, even_odd in zip(self.brange, self.even_odd):
#                dst = dsts[even_odd]
#                z = block_start + block_offset
#                bbox = bbox_lookup[z]
#                t = a.render(cm, src.path, vvote_field.path, dst, src_z=z, field_z=z, dst_z=z,
#                             bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
#                             mask_val=src_mask_val, mask_mip=src_mask_mip, prefix=prefix)
#                yield from t
#    ptask = []
#    start = time()
#    for irange, ieven_odd in zip(range_list, even_odd_list):
#        ptask.append(RenderTaskIteratorII(irange, ieven_odd))
#    
#    with ProcessPoolExecutor(max_workers=a.threads) as executor:
#        executor.map(remote_upload, ptask)
#   
#    end = time()
#    diff = end - start
#    print("Sending Render Tasks use time:", diff)
#    print('Run rendering')
#
#    start = time()
#    # wait 
#    a.wait_for_sqs_empty()
#    end = time()
#    diff = end - start
#    print("Executing Rendering use time:", diff)
#

