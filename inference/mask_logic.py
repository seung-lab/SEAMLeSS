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
  parser.add_argument('--paths', type=str, nargs='+')
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--op', type=str)
  parser.add_argument('--z_offsets', type=int, nargs='+',
    help='offsets from z to evaluate each image from paths')
  parser.add_argument('--dst_offset', type=int,
    help='offset from z where dst will be written')
  parser.add_argument('--mip_list', type=int, nargs='+',
    help='mip level of each image in paths')
  parser.add_argument('--dst_mip', type=int,
    help='mip level of dst image')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--pad',
    help='the size of the largest displacement expected; should be 2^high_mip',
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  args = parse_args(parser)
  # Only compute matches to previous sections
  a = get_aligner(args)
  a.device = torch.device('cpu')
  bbox = get_bbox(args)
  provenance = get_provenance(args)

  # Simplify var names
  mip = args.dst_mip
  mip_list = args.mip_list
  max_mip = args.max_mip
  pad = args.pad
  z_offsets = args.z_offsets
  dst_offset = args.dst_offset
  print('mip {}'.format(mip))
  print('mip_list {}'.format(mip_list))
  print('z_offsets {}'.format(z_offsets))
  print('dst_offset {}'.format(dst_offset))

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])

  # Create CloudVolume Manager
  cm = CloudManager(args.paths[0], max_mip, pad, provenance, batch_size=1,
                    size_chunk=256*4, batch_mip=mip)
  # Create src CloudVolumes
  cv_list = []
  for path in args.paths:
    cv = cm.create(path, data_type='uint8', num_channels=1,
                   fill_missing=True, overwrite=False)
    cv_list.append(cv.path)

  # Create dst CloudVolumes
  dst = cm.create(args.dst_path, data_type='uint8', num_channels=1, fill_missing=True,
                  overwrite=True).path

  prefix = str(mip)
  class TaskIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
              z_list = [z+zo for zo in z_offsets]
              dst_z = z + dst_offset
              t = a.mask_logic(cm, cv_list, dst, z_list, dst_z, bbox, mip_list, mip,
                               op=args.op)
              yield from t

  range_list = make_range(full_range, a.threads)

  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)

  start = time()
  ptask = []
  for i in range_list:
      ptask.append(TaskIterator(i))

  #with  LocalTaskQueue(parallel=1) as tq:
  #    tq.insert_all(ptask[0], args=[a])
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
  # wait

