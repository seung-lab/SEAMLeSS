import gevent.monkey
gevent.monkey.patch_all()
import torch

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue

import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from cloudvolume import CloudVolume
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
  parser.add_argument('--mip', type=int, default=0)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--threshold', type=int,
    help='Below this number, everything will be masked')
  parser.add_argument('--max_mip', type=int, default=4)
  args = parse_args(parser)
  # Only compute matches to previous sections
  # args.serial_operation = True
  a = get_aligner(args)
  a.device = torch.device('cpu')
  bbox = get_bbox(args)
  provenance = get_provenance(args)

  # Simplify var names
  mip = args.mip
  pad = 0
  threshold = args.threshold

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])

  src_cv = CloudVolume(args.src_path)
  info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='image',
    data_type='uint8',
    encoding="raw",
    resolution=src_cv.resolution,
    voxel_offset=src_cv.voxel_offset,
    chunk_size=src_cv.chunk_size,
    volume_size=src_cv.volume_size,
  )
  vol = CloudVolume(args.dst_path, info=info)
  vol.commit_info()

  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)
  batch =[]
  prefix = str(mip)
  class TaskIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
              t = a.threshold_and_mask(args.src_path, src_cv, args.dst_path, z, mip, bbox, threshold, args.max_mip)
              yield from t
  range_list = make_range(full_range, a.threads)

  start = time()
  ptask = []
  for i in range_list:
      ptask.append(TaskIterator(i))

  if a.distributed:
      with ProcessPoolExecutor(max_workers=a.threads) as executor:
          executor.map(remote_upload, ptask)
  else:
      for t in ptask:
          tq = LocalTaskQueue(parallel=1)
          tq.insert_all(t, args= [a])

  end = time()
  diff = end - start
  print("Sending Tasks use time:", diff)
  print('Running Tasks')
  start = time()
  if a.distributed:
    a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Tasks use time:", diff)
