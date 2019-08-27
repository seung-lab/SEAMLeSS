from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue, MockTaskQueue

import sys
import torch
import json
import math
import csv
from copy import deepcopy
from time import time, sleep
from as_args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from itertools import compress
from tasks import run
from boundingbox import BoundingBox
import numpy as np

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

def remote_upload_it(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

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
    help='relative path to CSV file identifying params to use per z range')
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
  print('Create src & align image CloudVolumes')
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  src_mask_cv = None
  tgt_mask_cv = None
  if args.src_mask_path:
    src_mask_cv = cm.create(args.src_mask_path, data_type='uint8', num_channels=1,
                               fill_missing=True, overwrite=False).path
    tgt_mask_cv = src_mask_cv

  if src_mask_cv != None:
      src_mask_cv = src_mask_cv.path
  if tgt_mask_cv != None:
      tgt_mask_cv = tgt_mask_cv.path

  # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius 
  block_dsts = {}
  block_types = ['even', 'odd']
  for i, block_type in enumerate(block_types):
    block_dst = cm.create(join(args.dst_path, 'image_blocks', block_type),
                    data_type='uint8', num_channels=1, fill_missing=True,
                    overwrite=True)
    block_dsts[i] = block_dst
  
  # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
  bbox_lookup = {}
  model_lookup = {}
  tgt_radius_lookup = {}
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
           tgt_radius_lookup[z] = tgt_radius
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
  initial_block_starts = list(range(args.z_start, args.z_stop, block_size))
  if initial_block_starts[-1] != args.z_stop:
    initial_block_starts.append(args.z_stop)
  block_starts = []
  for bs, be in zip(initial_block_starts[:-1], initial_block_starts[1:]):
    while bs in skip_list:
      bs += 1
      assert(bs < be)
    block_starts.append(bs)
  block_stops = block_starts[1:]
  if block_starts[-1] != args.z_stop:
    block_stops.append(args.z_stop)

  block_dst_lookup = {}
  block_start_lookup = {}
  starter_dst_lookup = {}
  copy_offset_to_z_range = {0: deepcopy(block_starts)}
  overlap_copy_range = set()
  starter_offset_to_z_range = {i: set() for i in range(min_offset, 0)}
  block_offset_to_z_range = {i: set() for i in range(1, block_size+10)} #TODO: Set the padding based on max(be-bs)
  # Reverse lookup to easily identify tgt_z for each starter z
  starter_z_to_offset = {} 
  for k, (bs, be) in enumerate(zip(block_starts, block_stops)):
    even_odd = k % 2
    for i, z in enumerate(range(bs, be+1)):
      if i > 0:
        block_start_lookup[z] = bs
        block_dst_lookup[z] = block_dsts[even_odd]
        if z not in skip_list:
          #print("in block_offset_to_z_range i", i, "z", z)
          block_offset_to_z_range[i].add(z)
          for tgt_offset in vvote_lookup[z]:
            tgt_z = z + tgt_offset
            if tgt_z <= bs:
              starter_dst_lookup[tgt_z] = block_dsts[even_odd]
              # ignore first block for stitching operations
              if k > 0:
                overlap_copy_range.add(tgt_z)
            if tgt_z < bs:
              starter_z_to_offset[tgt_z] = bs - tgt_z
              starter_offset_to_z_range[tgt_z - bs].add(tgt_z)
  offset_range = [i for i in range(min_offset, abs(min_offset)+1)]
  # check for restart
  print('Align starting from OFFSET {}'.format(args.restart))
  starter_restart = -100 
  if args.restart <= 0:
    starter_restart = args.restart 
  copy_offset_to_z_range = {k:v for k,v in copy_offset_to_z_range.items() 
                                              if k == args.restart}
  starter_offset_to_z_range = {k:v for k,v in starter_offset_to_z_range.items() 
                                              if k <= starter_restart}
  block_offset_to_z_range = {k:v for k,v in block_offset_to_z_range.items() 
                                              if k >= args.restart}
  # print('copy_offset_to_z_range {}'.format(copy_offset_to_z_range))
  # print('starter_offset_to_z_range {}'.format(starter_offset_to_z_range))
  # print('block_offset_to_z_range {}'.format(block_offset_to_z_range))
  # print('offset_range {}'.format(offset_range))
  copy_range = [z for z_range in copy_offset_to_z_range.values() for z in z_range]
  starter_range = [z for z_range in starter_offset_to_z_range.values() for z in z_range]


  # Create field CloudVolumes
  print('Creating field & overlap CloudVolumes')
  block_pair_fields = {}
  for z_offset in offset_range:
    block_pair_fields[z_offset] = cm.create(join(args.dst_path, 'field', 'block', 
                                                 str(z_offset)), 
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True)
  block_vvote_field = cm.create(join(args.dst_path, 'field', 'vvote'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)
  stitch_pair_fields = {}
  broadcasting_field = cm.create(join(args.dst_path, 'field', 
                                      'stitch', 'broadcasting'),
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=True)

  bs = block_starts[1]
  be = block_stops[1]
  block_range = range(bs, be+1)

  overlap_copy_range = set()
  for z in block_range:
      if z not in skip_list:
          for tgt_offset in vvote_lookup[z]:
              tgt_z = z + tgt_offset
              if tgt_z <= bs:
                  overlap_copy_range.add(tgt_z)
  overlap_copy_range =sorted(list(overlap_copy_range))
  #for bs, be in zip(block_starts[1:], block_stops[1:]):
  max_offset = 0
  stitch_offset_to_z_range =[]
  block_start_to_stitch_offsets = []
  for z in block_range[1:]:
      if z not in skip_list:
          max_offset = max(max_offset, tgt_radius_lookup[z])
          if len(block_start_to_stitch_offsets) < max_offset:
              stitch_offset_to_z_range.append(z)
              block_start_to_stitch_offsets.append(bs - z)
          else:
              break

  chunk_grid = a.get_chunk_grid(cm, bbox, mip, 0, 1000, pad)
  dst = block_dsts[0]
  serial_fields = block_pair_fields[0]
  block_start = block_starts[0]
  block_stop = block_stops[0]
#  a.new_align(src, dst, serial_fields, block_vvote_field, chunk_grid, mip, pad,
#              block_start, block_stop, chunk_size, args.param_lookup,
#              src_mask_cv=src_mask_cv,
#              src_mask_mip=src_mask_mip, src_mask_val=src_mask_val)


  src_cv = block_dst_lookup[bs+1]
  tgt_cv = block_dst_lookup[bs]
  a.get_stitch_field(model_lookup, src_cv, tgt_cv, block_vvote_field,
                     broadcasting_field, src, overlap_copy_range,
                     stitch_offset_to_z_range, mip, chunk_grid[0],
                     chunk_size, pad,
                     softmin_temp=2**mip, blur_sigma=1)

