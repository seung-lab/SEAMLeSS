import gevent.monkey
gevent.monkey.patch_all()

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
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from itertools import compress
from tasks import run
from boundingbox import BoundingBox
from cloudvolume import CloudVolume
import numpy as np

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
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--z_start', type=int)
  parser.add_argument('--z_stop', type=int)
  parser.add_argument('--max_mip', type=int, default=7)
  args = parse_args(parser)
  # Only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, 0, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)

  # Create src CloudVolumes
  print('Create src & align image CloudVolumes')
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False).path
  dst = cm.create(args.dst_path, 
                  data_type='uint8', num_channels=1, fill_missing=True, 
                  overwrite=True).path
  x_start = 0
  x_stop = 270336
  y_start = 0
  y_stop = 131072
  bbox_mip = 0
  bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)

  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)  

  def execute(task_iterator, z_range):
    if len(z_range) > 0:
      ptask = []
      range_list = make_range(z_range, a.threads)
      start = time()

      for irange in range_list:
          ptask.append(task_iterator(irange))
      if args.dry_run:
        for t in ptask:
         tq = MockTaskQueue(parallel=1)
         tq.insert_all(t, args=[a])
      else:
        if a.distributed:
          with ProcessPoolExecutor(max_workers=a.threads) as executor:
              executor.map(remote_upload, ptask)
        else:
          for t in ptask:
           tq = LocalTaskQueue(parallel=1)
           tq.insert_all(t, args=[a])
 
      end = time()
      diff = end - start
      print('Sending {} use time: {}'.format(task_iterator, diff))
      if a.distributed:
        print('Run {}'.format(task_iterator))
        # wait
        start = time()
        a.wait_for_sqs_empty()
        end = time()
        diff = end - start
        print('Executing {} use time: {}\n'.format(task_iterator, diff))

  # Task Scheduling Iterators
  print('Creating task scheduling iterators')
  cv = CloudVolume(dst, fill_missing=True, mip=max_mip)
  class StarterCopy():
    def __init__(self, z_range):
      print(z_range)
      self.z_range = z_range

    def __iter__(self):
      for z in self.z_range:
        test = cv[:,:,z // 100]
        if np.sum(test) == 0:
          for cur_mip in range(mip, max_mip+1):
            t =  a.copy(cm, src, dst, z, z // 100, bbox, cur_mip, is_field=False)
            yield from t 

  copy_range = range(args.z_start, args.z_stop, 100)
  print('COPY STARTING SECTION OF ALL BLOCKS')
  execute(StarterCopy, copy_range)

