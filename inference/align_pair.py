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

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--model_path', type=str,
    help='relative path to the ModelArchive to use for computing fields')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--src_z', type=int)
  parser.add_argument('--tgt_z', type=int)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0)
  parser.add_argument('--pad',
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  args = parse_args(parser)
  # Only compute matches to previous sections
  a = get_aligner(args)
  provenance = get_provenance(args)
  chunk_size = 1024
  model_path = args.model_path
  bbox = get_bbox(args)
  src_z = args.src_z
  tgt_z = args.tgt_z

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad
  src_mask_val = args.src_mask_val
  src_mask_mip = args.src_mask_mip

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)

  # Create src CloudVolumes
  print('Create src & align image CloudVolumes')
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False).path
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

  field = cm.create(join(args.dst_path, 'field'), data_type='int16', num_channels=2,
                         fill_missing=True, overwrite=True).path
  dst = cm.create(args.dst_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=True).path
  # Task scheduling functions
  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

  def execute(task_iterator, z_range):
    if len(z_range) > 0:
      ptask = []
      range_list = [z_range]
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
  class Copy():
    def __init__(self, z_range):
      print(z_range)
      self.z_range = z_range

    def __iter__(self):
      for z in self.z_range:
        t =  a.copy(cm, src, dst, z, z, bbox, mip, is_field=False,
                    mask_cv=src_mask_cv, mask_mip=src_mask_mip, mask_val=src_mask_val)
        yield from t

  class ComputeField(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __iter__(self):
      for z in self.z_range:
        t = a.compute_field(cm, model_path, src, dst, field,
                            z, tgt_z, bbox, mip, pad, src_mask_cv=src_mask_cv,
                            src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                            tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip,
                            tgt_mask_val=src_mask_val, prev_field_cv=None,
                            prev_field_z=None)
        yield from t

  class Render(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __iter__(self):
      for z in self.z_range:
        t = a.render(cm, src, field, dst, src_z=z, field_z=z, dst_z=z,
                     bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
                     mask_val=src_mask_val, mask_mip=src_mask_mip)
        yield from t

  # Serial alignment with block stitching 
  print('COPY TARGET SECTION')
  execute(Copy, [args.tgt_z])
  print('ALIGN SOURCE SECTION')
  execute(ComputeField, [args.src_z])
  execute(Render, [args.src_z])
