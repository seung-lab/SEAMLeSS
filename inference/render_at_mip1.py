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
from tasks import run
from boundingbox import BoundingBox
import numpy as np

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
  parser.add_argument('--downsample_shift', type=int, default=0,
    help='temporary hack to account for half pixel shifts caused by downsampling')
  parser.add_argument('--affine_lookup', type=str, 
    help='path to csv of affine transforms indexed by section')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--field_path', type=str)
  parser.add_argument('--fine_field_path', type=str)
  parser.add_argument('--coarse_field_path', type=str)
  parser.add_argument('--fine_mip', type=int)
  parser.add_argument('--coarse_mip', type=int)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--src_mip', type=int)
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
  parser.add_argument('--task_limit', type=int, default=500000,
    help='no. of tasks scheduled before a wait is put in place')
  args = parse_args(parser)
  # only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  src_mip = args.src_mip
  f_mip = args.fine_mip
  g_mip = args.coarse_mip
  max_mip = args.max_mip
  pad = args.pad

  # Compile ranges
  z_range = range(args.bbox_start[2], args.bbox_stop[2])

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=src_mip)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  f_field = cm.create(args.fine_field_path, data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)
  dst = cm.create(args.dst_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=True)
  g_field = cm.create(args.coarse_field_path, data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)
  field = cm.create(args.field_path, data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True)

  # compile model lookup per z index
  affine_lookup = None
  if args.affine_lookup:
    affine_lookup = {}
    with open(args.affine_lookup) as f:
      affine_list = json.load(f)
      for aff in affine_list:
        z = aff['z']
        affine_lookup[z] = np.array(aff['transform'])
        affine_lookup[z][:, 2] += args.downsample_shift

  prefix = ''

  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)


  class ComposeTaskIterator(object):
      def __init__(self, zrange):
          self.zrange = zrange
      def __iter__(self):
          print("range is ", self.zrange)
          for z in self.zrange:
              affine = None
              t = a.cloud_compose_field(cm, f_field.path, g_field.path,
                      field.path, z, z, z, bbox, f_mip,
                      g_mip, src_mip, affine, pad, prefix=prefix)
              yield from t

  ptask = []
  range_list = make_range(z_range, a.threads)
  
  start = time()
  for irange in range_list:
      ptask.append(ComposeTaskIterator(irange))
  
  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)
 
  end = time()
  diff = end - start
  print("Sending Compose Tasks use time:", diff)
  print('Running Compose Tasks')
  # wait 
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Compose Tasks use time:", diff)
 
  class RenderTaskIterator(object):
      def __init__(self, zrange):
        self.zrange = zrange
      def __iter__(self):
        print("range is ", self.zrange)
        for z in self.zrange:
          affine = None
          if affine_lookup:
            try:
                affine = affine_lookup[z]
            except KeyError:
                affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) 
          t = a.render(cm, src.path, field.path, dst.path, z, z, z, bbox,
                           src_mip, src_mip, affine=affine, prefix=prefix) 
          yield from t
 
  ptask = []
  start = time()
  for irange in range_list:
      ptask.append(RenderTaskIterator(irange))

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)

  end = time()
  diff = end - start
  print("Sending Render Tasks use time:", diff)
  print('Running Render Tasks')
  # wait 
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Render Tasks use time:", diff)


# # a.downsample_range(dst_cv, z_range, bbox, a.render_low_mip, a.render_high_mip)
