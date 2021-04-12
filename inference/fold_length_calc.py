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
  parser.add_argument('--mip', type=int)
  parser.add_argument('--thr_binarize', type=float, default=0,
    help='Threshold for binary mask')
  parser.add_argument('--w_connect', type=int, default=0,
    help='Width to dilate to connect adjacent components')
  parser.add_argument('--thr_filter', type=int, default=0,
    help='Size threshold to filter small components')
  parser.add_argument('--w_dilate', type=int, default=0,
    help='Width to dilate')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--chunk_size', nargs=2, type=int,
    help='chunk size')
  parser.add_argument('--pad', type=int, default=256,
    help='chunk padding')
  parser.add_argument('--return_skeleys', action='store_true')
  parser.add_argument('--longest_fold_cv', type=str, default=None)
  parser.add_argument('--count_pixels', action='store_true')
  args = parse_args(parser)
  # Only compute matches to previous sections
  # args.serial_operation = True
  a = get_aligner(args)
  a.device = torch.device('cpu')
  bbox = get_bbox(args)
  provenance = get_provenance(args)

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  chunk_pad = 0

  thr_binarize = args.thr_binarize
  w_connect = args.w_connect
  thr_filter = args.thr_filter
  w_dilate = args.w_dilate
  chunk_size = args.chunk_size
  pad = args.pad
  longest_fold_cv = args.longest_fold_cv

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, chunk_pad, provenance, batch_size=1,
                    size_chunk=chunk_size[0], batch_mip=mip)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)

  info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='segmentation',
    data_type='uint32',
    encoding="compressed_segmentation",
    resolution=cm.info['scales'][mip]['resolution'],
    voxel_offset=cm.dst_voxel_offsets[mip],
    chunk_size=[*cm.dst_chunk_sizes[mip], 1],
    volume_size=cm.vec_total_sizes[mip],
  )
  vol = CloudVolume(args.dst_path, info=info)
  vol.commit_info()

  if longest_fold_cv is not None:
    longest_fold_info = CloudVolume.create_new_info(
      num_channels=1,
      layer_type='segmentation',
      data_type='uint32',
      encoding="compressed_segmentation",
      resolution=cm.info['scales'][mip]['resolution'],
      voxel_offset=[0,0,0],
      chunk_size=[1,1,1],
      volume_size=cm.vec_total_sizes[mip],
    )
    longest_fold_vol = CloudVolume(longest_fold_cv, info=longest_fold_info)
    longest_fold_vol.commit_info()

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
              t = a.calculate_fold_lengths(cm, src.path, args.dst_path, z, mip, bbox, chunk_size, pad, thr_binarize, w_connect, 
                                    thr_filter, args.return_skeleys, longest_fold_cv, args.count_pixels)
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