import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue

import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from time import time
from tasks import run 

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

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--src_mip', type=int)
  parser.add_argument('--dst_mip', type=int)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  # parser.add_argument('--save_intermediary', action='store_true')
  args = parse_args(parser)
  args.max_mip = args.dst_mip
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  src_mip = args.src_mip
  dst_mip = args.dst_mip
  a.chunk_size = (1024, 1024)
  pad = 0
  print('src_mip {}'.format(src_mip))
  print('dst_mip {}'.format(dst_mip))
  print('chunk_size {}'.format(chunk_size))

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, dst_mip, pad, provenance, batch_size=1,
                    size_chunk=1, batch_mip=dst_mip)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False).path

  seam_dir = 'seams/{}_{}'.format(src_mip, dst_mip)
  # Create dst CloudVolumes
  dst_pre = cm.create(join(args.dst_path, seam_dir, 'pre'),
                  data_type='float32', num_channels=1, fill_missing=True,
                  overwrite=True).path
  dst_post = cm.create(join(args.dst_path, seam_dir, 'post'),
                  data_type='float32', num_channels=1, fill_missing=True,
                  overwrite=True).path

  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)

  class ComputeSmoothnessIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
            t = a.compute_smoothness(cm, src, dst_pre, z, z, bbox, 
                                     src_mip)
            yield from t

  range_list = make_range(full_range, a.threads)

  start = time()
  ptask = []
  for i in range_list:
      ptask.append(ComputeSmoothnessIterator(i))

  if a.distributed:
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
  else:
      for t in ptask:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(t, args= [a])

  end = time()
  diff = end - start
  print("Sending ComputeSmoothness use time:", diff)
  start = time()
  print('Running Tasks')
  if a.distributed:
    a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("runtime:", diff)

  class SumPoolIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
            t = a.sum_pool(cm, dst_pre, dst_post, z, z, bbox, src_mip, dst_mip)
            yield from t

  start = time()
  ptask = []
  for i in range_list:
      ptask.append(SumPoolIterator(i))

  if a.distributed:
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
  else:
      for t in ptask:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(t, args= [a])

  end = time()
  diff = end - start
  print("Sending SumPoolTasks use time:", diff)
  start = time()
  print('Running Tasks')
  if a.distributed:
    a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("runtime:", diff)

