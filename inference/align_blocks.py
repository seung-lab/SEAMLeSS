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
  parser.add_argument('--param_lookup', type=str,
    help='relative path to CSV file identifying model to use per z range')
  # parser.add_argument('--z_range_path', type=str, 
  #   help='path to csv file with list of z indices to use')
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
  parser.add_argument('--overlap', type=int, default=3,
    help='int for number of sections to include in vector voting')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--restart', type=int, default=0)
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
  block_size = args.block_size

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)

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
  
  # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
  bbox_lookup = {}
  model_lookup = {}
  vvote_lookup = {}
  skip_list = [] 
  with open(args.param_lookup) as f:
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
         tgt_radius = int(r[8])
         skip = bool(int(r[9]))
         bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
         # print('{},{}'.format(z_start, z_stop))
         for z in range(z_start, z_stop):
           if skip:
             skip_list.append(z)
           bbox_lookup[z] = bbox 
           model_lookup[z] = model_path
           vvote_lookup[z] = [-i for i in range(1, tgt_radius+1)]

  # Filter out skipped sections from vvote_offsets
  min_offset = 0
  for z, tgt_radius in vvote_lookup.items():
    offset = 0
    for i, r in enumerate(tgt_radius):
      while r + offset + z in skip_list:
        offset -= 1
      tgt_radius[i] = r + offset
    min_offset = min(min_offset, r + offset)
    offset = 0 
    vvote_lookup[z] = tgt_radius

  # Adjust block starts so they don't start on a skipped section
  initial_block_starts = range(args.z_start, args.z_stop, block_size)
  block_starts = []
  for bs, be in zip(initial_block_starts[:-1], initial_block_starts[1:]):
    while bs in skip_list:
      bs += 1
      assert(bs < be)
    block_starts.append(bs)
  # Assign even/odd to each block start so results are stored in appropriate CloudVolume
  # Block offsets at 0 are all copied
  # Block offsets < 0 are all aligned without vvote to section at offset 0
  # Block offsets > 0 are all aligned with vvote to previous sections
  dst_lookup = {}
  copy_dict = {0: deepcopy(block_starts)}
  starter_dict = {i: [] for i in range(min_offset, 0)}
  starter_lookup = {} 
  block_range = {i: [] for i in range(1, block_size)}
  block_range_no_skips = {i: [] for i in range(1, block_size)}
  for k, (bs, be) in enumerate(zip(block_starts[:-1], block_starts[1:])):
    even_odd = k % 2
    for i, z in enumerate(range(bs, be)):
      block_range_no_skips[i] = z
      dst_lookup[z] = dsts[even_odd]
      if z not in skip_list:
        block_range[i] = z
        for tgt_offset in vvote_lookup[z]:
          tgt_z = z + tgt_offset
          if tgt_z < bs:
            starter_lookup[tgt_z] = tgt_z - bs
            starter_dict[tgt_z - bs].append(tgt_z)
  offset_range = [i for i in range(min_offset, abs(min_offset)+1)]
  # check for restart
  print('Starting from OFFSET {}'.format(args.restart))
  copy_dict = {k:v for k,v in copy_dict if k == args.restart}
  starter_dict = {k:v for k,v in starter_dict if k <= args.restart}
  block_range = {k:v for k,v in block_range if k >= args.restart}
  print('copy_dict {}'.format(copy_dict))
  print('starter_dict {}'.format(starter_dict))
  print('block_range {}'.format(block_range))
  print('offset_range {}'.format(offset_range))
  copy_list = [*v for k,v in copy_dict.items()]
  starter_list = [*v for k,v in starter_dict.items()]

  # if args.z_range_path:
  #   print('Compiling z_range from {}'.format(args.z_range_path))
  #   block_endpoints = range(args.z_start, args.z_stop+args.block_size, args.block_size)
  #   block_pairs = list(zip(block_endpoints[:-1], block_endpoints[1:]))
  #   tmp_block_range = []
  #   tmp_even_odd_range = []
  #   with open(args.z_range_path) as f:
  #     reader = csv.reader(f, delimiter=',')
  #     for k, r in enumerate(reader):
  #        if k != 0:
  #          z_pair = int(r[0]), int(r[1])
  #          print('Filtering block_range by {}'.format(z_pair))
  #          block_filter = [ranges_overlap(z_pair, b_pair) for b_pair in block_pairs]
  #          affected_blocks = list(compress(block_range, block_filter))
  #          affected_even_odd = list(compress(even_odd_range, block_filter))
  #          print('Affected block_starts {}'.format(affected_blocks))
  #          tmp_block_range.extend(affected_blocks)
  #          tmp_even_odd_range.extend(affected_even_odd)
  #   block_range = tmp_block_range
  #   even_odd_range = tmp_even_odd_range
  # print('block_range {}'.format(block_range))
  # print('even_odd_range {}'.format(even_odd_range))


  # Create field CloudVolumes
  pair_fields = {}
  for z_offset in offset_range:
    pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True).path
  vvote_field = cm.create(join(args.dst_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  # Task scheduling functions
  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)  

  def execute(z_range, task_iterator, task_name):
    ptask = []
    range_list = make_range(z_range, a.threads)
    start = time()
    for irange in range_list:
        ptask.append(task_iterator(irange))
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
    end = time()
    diff = end - start
    print('Sending {} use time: {}'.format(task_name, diff))
    print('Run {}'.format(task_name)
    # wait
    start = time()
    a.wait_for_sqs_empty()
    end = time()
    diff = end - start
    print('Executing {} use time: {}'.format(task_name, diff))

  # Task Scheduling Iterators
  class CopyTaskIterator():
    def __init__(self, z_range):
      self.z_range = z_range

    def __repr__(self):
      return 'CopyTask'
 
    def __iter__(self):
      for z in self.z_range:
        dst = dst_lookup[z]
        bbox = bbox_lookup[z]
        t =  a.copy(cm, src.path, dst, z, z, bbox, mip, is_field=False,
                    mask_cv=src_mask_cv, mask_mip=src_mask_mip, mask_val=src_mask_val)
        yield from t 

  class ComputeFieldTaskIterator(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __repr__(self):
      return 'ComputeFieldTask'

    def __iter__(self):
      for src_z in self.z_range:
        dst = dst_lookup[src_z]
        model_path = model_lookup[src_z]
        bbox = bbox_lookup[src_z]
        z_offset = starter_lookup[src_z]
        field = pair_fields[z_offset]
        tgt_z = src_z + z_offset
        t = a.compute_field(cm, model_path, src.path, dst, field, 
                            src_z, tgt_z, bbox, mip, pad, src_mask_cv=src_mask_cv,
                            src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                            tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                            tgt_mask_val=src_mask_val, prev_field_cv=None, 
                            prev_field_z=None)
        yield from t

  class RenderTaskIterator(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __repr__(self):
      return 'RenderTask'

    def __iter__(self):
      for z in self.z_range:
        dst = dst_lookup[z]
        bbox = bbox_lookup[z]
        t = a.render(cm, src.path, serial_field, dst, src_z=z, field_z=z, dst_z=z,
                     bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
                     mask_val=src_mask_val, mask_mip=src_mask_mip)
        yield from t

  class ComputeFieldTaskIteratorII(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __repr__(self):
      return 'ComputeFieldTask'

    def __iter__(self):
      for src_z in self.z_range:
        dst = dst_lookup[src_z]
        bbox = bbox_lookup[src_z]
        model_path = model_lookup[src_z]
        tgt_offsets = vvote_lookup[src_z]
        for tgt_offset in tgt_offsets:
          tgt_z = src_z + tgt_offset
          field = pair_fields[tgt_offset]
          t = a.compute_field(cm, model_path, src.path, dst, field, 
                              src_z, tgt_z, bbox, mip, pad, src_mask_cv=src_mask_cv,
                              src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                              tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                              tgt_mask_val=src_mask_val, prev_field_cv=vvote_field.path, 
                              prev_field_z=tgt_z)
          yield from t

  class VectorVoteTaskIterator(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __repr__(self):
      return 'VectorVoteTask'

    def __iter__(self):
      for z in self.z_range
        bbox = bbox_lookup[z]
        tgt_offsets = vvote_lookup[z]
        fields = {i: pair_fields[i] for i in tgt_offsets}
        t = a.vector_vote(cm, fields, vvote_field.path, z, bbox,
                          mip, inverse=False, serial=True, 
                          softmin_temp=2**mip, blur_sigma=1)
        yield from t

  class RenderTaskIteratorII(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __repr__(self):
      return 'RenderTask' 

    def __iter__(self):
      for z in self.z_range:
        dst = dst_lookup[z]
        bbox = bbox_lookup[z]
        t = a.render(cm, src.path, vvote_field.path, dst, src_z=z, field_z=z, dst_z=z,
                     bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
                     mask_val=src_mask_val, mask_mip=src_mask_mip)
        yield from t


  # Serial alignment script
  print('COPY STARTING SECTION OF ALL BLOCKS')
  execute(CopyTaskIterator, copy_list)
  print('ALIGN STARTER SECTIONS FOR EACH BLOCK')
  execute(ComputeFieldTaskIterator, starter_list)
  execute(RenderTaskIterator, starter_list)
  for z_offset, z_range in block_range:
    print('ALIGN BLOCK OFFSET {}'.format(z_offset))
    execute(ComputeFieldTaskIteratorII, z_range)
    execute(VectorVoteTaskIterator, z_range)
    execute(RenderTaskIteratorII, z_range)

