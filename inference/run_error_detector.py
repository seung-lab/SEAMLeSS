import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue

import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox_3d, get_provenance
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
  parser.add_argument('--model_path', type=str,
    help='Relative path to the ModelArchive to use for error detection')
  parser.add_argument('--src_img_path', type=str,
    help='CloudVolume path of source EM image')
  parser.add_argument('--src_seg_path', type=str,
    help='CloudVolume path of source flat segmentation')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path of error map destination')
  parser.add_argument('--mip', type=int,
    help='Mip level of error detection')
  parser.add_argument('--chunk_size', nargs=3, type=int,
    help='Chunk size in designated mip level')
  parser.add_argument('--patch_size', nargs=3, type=int,
    help='Input sample size in designated mip level')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--max_displacement', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  
  args = parse_args(parser)
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox_3d(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.max_displacement
  patch_size = args.patch_size
  chunk_size = args.chunk_size
  # chunk_size = [512,512,64]

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2], chunk_size[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_img_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size[0], batch_mip=mip)

  # Create src CloudVolumes
  src_img = cm.create(args.src_img_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  src_seg = cm.create(args.src_seg_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)

  # Create dst CloudVolumes
  dst = cm.create(args.dst_path,
                  data_type='uint8', num_channels=1, fill_missing=True,
                  overwrite=True)

  
  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)
  batch =[]
  prefix = str(mip)
  class TaskIterator():
      def __init__(self):
          pass
      def __iter__(self):
          t = a.error_detect_volume(cm, args.model_path, src_seg.path, src_img.path, dst.path, mip, bbox, chunk_size, patch_size, str(mip))
          yield from t
  # range_list = make_range(full_range, a.threads)

  start = time()
  ptask = []
  ptask.append(TaskIterator())


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
  start = time()
  #if args.use_sqs_wait:
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Tasks use time:", diff)
