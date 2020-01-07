"""
Stitch even and odd blocks together
T Macrina
190305

TODO: This script does not properly handle the last block (reverse compose)
      or the first block (forward compose). Proper handling should be put
      in place.
"""
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

def interpolate(x, start, stop_dist):
  """Return interpolation value of x for range(start, stop)

  Args
     x: int location
     start: location corresponding to 1
     stop_dist: distance from start corresponding to 0 
  """
  assert(stop_dist != 0)
  stop = start + stop_dist
  d = (stop - x) / (stop - start) 
  return min(max(d, 0.), 1.)

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--param_lookup', type=str,
    help='relative path to CSV file identifying params to use per z range')
  # parser.add_argument('--z_range_path', type=str, 
  #   help='path to csv file with list of z indices to use')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--z_start', type=int)
  parser.add_argument('--z_stop', type=int)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--decay_dist', type=int, default=10)
  parser.add_argument('--suffix', type=str, default='',
    help='string to append to directory names')
  parser.add_argument(
    "--skip_stitching",
    action='store_true',
    help="If True, skip compute field and vector voting"
  )
  parser.add_argument(
    "--skip_render",
    action='store_true',
    help="If True, skip rendering"
  )
  parser.add_argument(
    "--render_mip",
    type=int
  )
  args = parse_args(parser)
  # Only compute matches to previous sections
  a = get_aligner(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.pad
  src_mask_val = 1
  src_mask_mip = 8
  block_size = args.block_size
  do_stitching = not args.skip_stitching
  do_render = not args.skip_render
  render_mip = args.render_mip or args.mip

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)

  cmr = CloudManager(
    args.src_path,
    max_mip,
    pad,
    provenance,
    batch_size=1,
    size_chunk=chunk_size,
    batch_mip=render_mip,
  )
  
  # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
  bbox_lookup = {}
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
  print('block_starts {}'.format(block_starts))

  # Compile ranges
  decay_dist = args.decay_dist
  compose_range = range(args.z_start, args.z_stop)
  influencing_blocks_lookup = {z: [] for z in compose_range}
  for b_start in block_starts:
    for z in range(b_start+1, b_start+decay_dist+1):
      if z < args.z_stop:
        influencing_blocks_lookup[z].append(b_start)

  # Create CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                  fill_missing=True, overwrite=False).path
  src_mask_cv = None
  tgt_mask_cv = None

  broadcasting_field = cm.create(join(args.dst_path, 'field', 
                                      'stitch3', 'broadcasting'),
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=False).path
  block_field = cm.create(join(args.dst_path, 'field', 'vvote'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False).path

  compose_field = cm.create(join(args.dst_path, 'field', 'stitch3{}'.format(args.suffix), 
                                 'compose'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=do_stitching).path
  final_dst = cmr.create(join(args.dst_path, 'image_stitch3{}'.format(args.suffix)), 
                        data_type='uint8', num_channels=1, fill_missing=True, 
                        overwrite=do_render).path

  # Task scheduling functions
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
        if do_stitching:
          a.wait_for_sqs_empty()
        end = time()
        diff = end - start
        print('Executing {} use time: {}\n'.format(task_iterator, diff))

  # Task Scheduling Iterators
  class StitchCompose(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __iter__(self):
      for z in self.z_range:
        influencing_blocks = influencing_blocks_lookup[z]
        factors = [interpolate(z, bs, decay_dist) for bs in influencing_blocks]
        factors += [1.]
        print('z={}\ninfluencing_blocks {}\nfactors {}'.format(z, influencing_blocks, 
                                                               factors))
        bbox = bbox_lookup[z]
        cv_list = [broadcasting_field]*len(influencing_blocks) + [block_field]
        z_list = list(influencing_blocks) + [z]
        t = a.multi_compose(cm, cv_list, compose_field, z_list, z, bbox, 
                            mip, mip, factors, pad)
        yield from t

  class StitchRender(object):
    def __init__(self, z_range):
      self.z_range = z_range

    def __iter__(self): 
      for z in self.z_range:
        bbox = bbox_lookup[z]
        t = a.render(cm, src, compose_field, final_dst, src_z=z, field_z=z, dst_z=z, 
                     bbox=bbox, src_mip=mip, field_mip=mip)
        yield from t

  if do_stitching:
    execute(StitchCompose, compose_range)
  if do_render:
    execute(StitchRender, compose_range)
