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
  parser.add_argument('--tgt_radius', type=int, default=3,
    help='int for number of sections to include in vector voting')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--overwrite_dst', action='store_true')
  args = parse_args(parser)
  # Only compute matches to previous sections
  a = get_aligner(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad

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

  copy_range = full_range[overlap:]

  print('copy_range {}'.format(copy_range))

  # Create CloudVolume Manager
  info_path = join(args.src_path, 'image_blocks', 'even')
  cm = CloudManager(info_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)
  # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius 
  block_types = ['even', 'odd']
  srcs = {}
  for i, block_type in enumerate(block_types):
    src = cm.create(join(args.src_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=False)
    srcs[i] = src.path 
  dsts = {}
  for i, block_type in enumerate(block_types):
    dst = cm.create(join(args.dst_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=args.overwrite_dst)
    dsts[i] = dst.path 

  # Create field CloudVolumes
  src_field = cm.create(join(args.src_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)
  dst_field = cm.create(join(args.dst_path, 'field', 'vvote_{}'.format(overlap)), 
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=args.overwrite_dst)

  ###########################
  # Serial alignment script #
  ###########################
  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)  
 
  # class CopyTaskIteratorImage():
  #     def __init__(self, brange, even_odd):
  #         self.brange = brange
  #         self.even_odd = even_odd
  #     def __iter__(self):
  #         for block_offset in copy_range:
  #           prefix = block_offset
  #           for block_start, even_odd in zip(self.brange, self.even_odd):
  #             src = srcs[even_odd]
  #             dst = dsts[even_odd]
  #             z = block_start + block_offset
  #             bbox = bbox_lookup[z]
  #             print("src {}\n"
  #                   "dst {}\n"
  #                   "src_z {}, dst_z {}\n".format(src, dst, z, z))
  #             t =  a.copy(cm, src, dst, z, z, bbox, mip, is_field=False, prefix=prefix)
  #             yield from t 

  # print('Scheduling CopyTasksImage')
  # ptask = []
  range_list = make_range(block_range, a.threads)
  # even_odd_list = make_range(even_odd_range, a.threads)
  # print('range_list {}'.format(range_list))
  # print('even_odd_list {}'.format(even_odd_list))
  # 
  # start = time()
  # for irange, ieven_odd in zip(range_list, even_odd_list):
  #     ptask.append(CopyTaskIteratorImage(irange, ieven_odd))

  # with ProcessPoolExecutor(max_workers=a.threads) as executor:
  #     executor.map(remote_upload, ptask)
 
  # end = time()
  # diff = end - start
  # print("Sending CopyTasksImage use time:", diff)
  # print('Run CopyTasksImage; no waiting')
 
  class CopyTaskIteratorField():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for block_offset in copy_range:
            prefix = block_offset
            for block_start in self.brange:
              z = block_start + block_offset 
              bbox = bbox_lookup[z]
              print("src_field {}\n"
                    "dst_field {}\n"
                    "src_z {}, dst_z {}\n".format(src, dst, z, z))
              t = a.copy(cm, src_field, dst_field, z, z, bbox, mip, 
                         is_field=True, prefix=prefix) 
              yield from t

  print('Scheduling CopyTasksField')
  ptask = []
  start = time()
  for irange in range_list:
      ptask.append(CopyTaskIteratorField(irange))

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)
  end = time()
  diff = end - start
  print("Sending CopyTasksField use time:", diff)
  # wait
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing CopyTasksField use time:", diff) 
