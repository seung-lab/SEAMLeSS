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
from new_args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from itertools import compress
from tasks import run
from boundingbox import BoundingBox
import numpy as np
from resend_task import calc_start_z

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

def remote_upload_it(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

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
  parser.add_argument('--param_lookup', type=str,
    help='relative path to CSV file identifying params to use per z range')
  # parser.add_argument('--z_range_path', type=str, 
  #   help='path to csv file with list of z indices to use')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--src_mask_path', type=str, default='',
    help='CloudVolume path of mask to use with src images; default None')
  parser.add_argument('--src_mask_mip', type=int, default=8,
    help='MIP of source mask')
  parser.add_argument('--src_mask_val', type=int, default=1,
    help='Value of of mask that indicates DO NOT mask')
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--z_start', type=int)
  parser.add_argument('--z_stop', type=int)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--extra_off', type=int, default=None)
  parser.add_argument('--pad',
    help='the size of the largest displacement expected; should be 2^high_mip',
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--restart', type=int, default=0)
  args = parse_args(parser)
  # Only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  provenance = get_provenance(args)

  chunk_size = 256

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad
  src_mask_val = args.src_mask_val
  src_mask_mip = args.src_mask_mip
  block_size = args.block_size
  timeout = args.IO_timeout
  extra_off = args.extra_off
  if extra_off is None:
      extra_off = pad
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

  # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius 
  block_dsts = {}
  block_types = ['even', 'odd']
  for i, block_type in enumerate(block_types):
    block_dst = cm.create(join(args.dst_path, 'image_blocks', block_type),
                    data_type='uint8', num_channels=1, fill_missing=True,
                    overwrite=True)
    block_dsts[i] = block_dst.path
  
  # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
  bbox_lookup = {}
  model_lookup = {}
  tgt_radius_lookup = {}
  vvote_lookup = {}
  skip_list = [] 
  with open(args.param_lookup) as f:
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
         tgt_radius = int(r[8])
         skip = bool(int(r[9]))
         bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
         # print('{},{}'.format(z_start, z_stop))
         for z in range(z_start, z_stop):
           if skip:
             skip_list.append(z)
           bbox_lookup[z] = bbox 
           model_lookup[z] = model_path
           tgt_radius_lookup[z] = tgt_radius
           vvote_lookup[z] = [-i for i in range(1, tgt_radius+1)]

  # Adjust block starts so they don't start on a skipped section
  initial_block_starts = list(range(args.z_start, args.z_stop, block_size))
  if initial_block_starts[-1] != args.z_stop:
    initial_block_starts.append(args.z_stop)
  block_starts = []
  for bs, be in zip(initial_block_starts[:-1], initial_block_starts[1:]):
    while bs in skip_list:
      bs += 1
      assert(bs < be)
    block_starts.append(bs)
  block_stops = block_starts[1:]
  if block_starts[-1] != args.z_stop:
    block_stops.append(args.z_stop)

  block_dst_lookup={}
  for k, (bs, be) in enumerate(zip(block_starts, block_stops)):
    even_odd = k % 2
    block_dst_lookup[bs] = even_odd

  # Create field CloudVolumes
  print('Creating field & overlap CloudVolumes')
  block_pair_field = cm.create(join(args.dst_path, 'field', 'block'),
                                      data_type='int16', num_channels=2,
                                      fill_missing=True, overwrite=True).path

  block_vvote_field = cm.create(join(args.dst_path, 'field', 'vvote'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True).path
  broadcasting_field = cm.create(join(args.dst_path, 'field',
                                      'stitch', 'broadcasting'),
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=True).path
  tmp_vvote_field_cv = cm.create(join(args.dst_path, 'field', 'vvote_tmp'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=True).path

  tmp_img_cv = cm.create(join(args.dst_path, 'image_blocks', 'tmp'),
                  data_type='uint8', num_channels=1, fill_missing=True,
                  overwrite=True).path

  pre_field_cv = cm.create(join(args.dst_path, 'field', 'pre_field_cv'),
                           data_type='int16', num_channels=2,
                           non_aligned_writes=True,
                           fill_missing=True, overwrite=True).path

  tmp_profile_cv = cm.create(join(args.dst_path, 'field', 'profile_tmp'),
                           data_type='int16', num_channels=2,
                           non_aligned_writes=True,
                           fill_missing=True, overwrite=True).path

  # Task scheduling functions
  def remote_upload_it(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

  rows = 14
  super_chunk_len = 4
  overlap_chunks = 2 * (super_chunk_len+1)

  chunk_grid = a.get_chunk_grid(cm, bbox, mip, overlap_chunks, rows, pad)
  #chunk_grid = a.get_chunk_grid(cm, bbox, mip, 0, 1000, pad)
  qu = args.queue_name
  #start_z = -1
  #start_zs =[-1] * len(block_starts)
  block_align_finish_dir = args.dst_path+"/image_blocks/finished/"+str(mip)+"/"
  block_align_task_finish_dir = block_vvote_field+"/block_alignment_done/{}/".format(str(mip))
  print("block_starts",len(block_starts), block_starts)
  print("block_stops", len(block_stops), block_stops)
  bstart_list, bend_list, start_z_list = calc_start_z(block_starts, block_stops,
                                    block_align_task_finish_dir,
                                    block_align_finish_dir,
                                    skip_list)
  class AlignT(object):
      def __init__(self, bs_list, be_list, start_z):
          self.bs_list = bs_list
          self.be_list = be_list
          self.start_z = start_z
          print("*********self bs_list is ", self.bs_list)
          print("*********be_list is  ", self.be_list)

      def __iter__(self):
          for i in range(len(self.be_list)):
              bs = self.bs_list[i]
              be = self.be_list[i]
              start  = self.start_z[i]
              even_odd = block_dst_lookup[bs]
              dst = block_dsts[even_odd]
              finish_dir = block_align_finish_dir+str(bs)+"/"
              t = a.new_align_task(bs, be+1, start, src, dst,
                                   block_pair_field,
                                   block_vvote_field,
                                   chunk_grid, mip, pad,
                                   chunk_size, args.param_lookup, qu,
                                   finish_dir, timeout, extra_off,
                                   pre_field_cv,
                                   src_mask_cv=src_mask_cv,
                                   src_mask_mip=src_mask_mip,
                                   src_mask_val=src_mask_val,
                                   super_chunk_len=super_chunk_len,
                                   overlap_chunks=overlap_chunks)
              yield from t

  #print("z_range is ", z_range)
  ptask = []
  bs_list = make_range(bstart_list, a.threads)
  be_list = make_range(bend_list, a.threads)
  start_list = make_range(start_z_list, a.threads)
  #bs_list = make_range(block_starts, a.threads)
  #be_list = make_range(block_stops, a.threads)
  #start_list = make_range(start_zs, a.threads)

  print("bs-list", bs_list)
  print("be-list", be_list)
  print("start-list", start_list)
  for i in range(len(bs_list)):
      bs = bs_list[i]
      be = be_list[i]
      start = start_list[i]
      ptask.append(AlignT(bs, be, start))
  for i  in ptask[0]:
      print(i)
  if len(bstart_list) !=0:
      with ProcessPoolExecutor(max_workers=a.threads) as executor:
          executor.map(remote_upload_it, ptask)
      start = time()
      #print("start until now time", start - begin_time)
      #a.wait_for_queue_empty(dst.path, 'load_image_done/{}'.format(mip), len(batch))
      a.wait_for_queue_empty(block_align_task_finish_dir, '',
                             len(bstart_list), 30)
      #a.wait_for_sqs_empty()

      end = time()
      diff = end - start
      print("Executing Loading Tasks use time:", diff)

  stitch_get_field_task_finish=broadcasting_field+'/get_stitch_field_done/{}/'.format(str(mip))
  stitch_get_field_slice_finish=broadcasting_field+'/finish_slice/'+str(mip)+'/'

  class StitchGetFieldT(object):
      def __init__(self, bs_list, be_list, start_z):
          self.bs_list = bs_list
          self.be_list = be_list
          self.start_z = start_z
          print("*********self bs_list is ", self.bs_list)
          print("*********be_list is  ", self.be_list)
      def __iter__(self):
          for i in range(len(self.be_list)):
              bs = self.bs_list[i]
              be = self.be_list[i]
              start_z  = self.start_z[i]
              even_odd = block_dst_lookup[bs]
              src_cv = block_dsts[even_odd]
              tgt_cv = block_dsts[(even_odd+1)%2]
              finish_dir = stitch_get_field_slice_finish+str(bs)+'/'
              t = a.stitch_get_field_task_generator(qu, args.param_lookup, bs,
                                                    be, src_cv, tgt_cv,
                                                    block_vvote_field,
                                                    broadcasting_field,
                                                    tmp_img_cv,
                                                    tmp_vvote_field_cv,
                                                    pre_field_cv,
                                                    tmp_profile_cv,
                                                    mip, start_z,
                                                    chunk_grid,
                                                    chunk_size, pad,
                                                    finish_dir, timeout,
                                                    extra_off, super_chunk_len,
                                                    2**mip, 1)
              yield from t

  #print("z_range is ", z_range)
  ptask = []
  bstart_list, bend_list, start_z_list = calc_start_z(block_starts[1:], block_stops[1:],
                                  stitch_get_field_task_finish,
                                  stitch_get_field_slice_finish,
                                  skip_list)

  bs_list = make_range(bstart_list, a.threads)
  be_list = make_range(bend_list, a.threads)
  start_list = make_range(start_z_list, a.threads)
  print("bs_list is ", bs_list)
  print("be_list is ", be_list)
  print("start_list is ", start_list)
  for i in range(len(bs_list)):
      bs = bs_list[i]
      be = be_list[i]
      start = start_list[i]
      ptask.append(StitchGetFieldT(bs, be, start))
  
  for i in ptask[0]:
      print(" in ptask", i)

  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload_it, ptask)

  start = time()
  #print("start until now time", start - begin_time)
  #a.wait_for_queue_empty(dst.path, 'load_image_done/{}'.format(mip), len(batch))
  #a.wait_for_sqs_empty()
  a.wait_for_queue_empty(stitch_get_field_task_finish, '',
                         len(bstart_list), 30)
  end = time()
  diff = end - start 
  print("Executing Loading Tasks use time:", diff)

