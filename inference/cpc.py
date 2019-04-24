import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue

import sys
import torch
import json
from time import time, sleep
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from tasks import run 

def print_run(diff, n_tasks):
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
  parser.add_argument('--src_info_path', type=str, default='',
    help='str to existing CloudVolume path to use as template for new CloudVolumes')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--src_mip', type=int,
    help='int for input MIP')
  parser.add_argument('--dst_mip', type=int,
    help='int for output MIP, which will dictate the size of the block used')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--z_offset', type=int, default=-1,
    help='int for offset of section to be compared against')
  parser.add_argument('--unnormalized', action='store_true', 
    help='do not normalize the CPC output, save as float')
  args = parse_args(parser)
  if args.src_info_path == '':
    args.src_info_path = args.src_path
  # Only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  max_mip = args.max_mip
  pad = args.pad

  # Compile ranges
  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_info_path, max_mip, pad, provenance)
  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False).path
  data_type = 'uint8'
  if args.unnormalized:
    data_type = 'float32'
  
  # Create dst CloudVolumes for each block, since blocks will overlap by 3 sections
  dst_path = join(args.src_path, 'cpc', '{}_{}'.format(args.src_mip, args.dst_mip),
                  '{}'.format(args.z_offset)) 
  if args.unnormalized:
    dst_path = join(args.src_path, 'cpc', 'unnormalized', 
                    '{}_{}'.format(args.src_mip, args.dst_mip),
                    '{}'.format(args.z_offset)) 
  dst = cm.create(dst_path, data_type=data_type, num_channels=1, fill_missing=True, 
                  overwrite=True).path

  ##############
  # CPC script #
  ##############
  class TaskIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
            #print("Fcorr for z={} and z={}".format(z, z+1))
            t = a.cpc(cm, src, src, dst, z, z+args.z_offset, bbox, 
                      args.src_mip, args.dst_mip, norm=not args.unnormalized)
            yield from t

  range_list = make_range(z_range, a.threads)

  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)

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
