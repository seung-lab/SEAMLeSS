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
from as_args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from itertools import compress
from tasks import run 
from boundingbox import BoundingBox
from tasks import remote_upload

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

  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=mip)
  
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
         #bbox_lookup[z] = bbox 

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
    #for z in range(b_start+1, b_start+decay_dist+1):
      z = b_start+1
      if z < args.z_stop:
          influencing_blocks_lookup[z].append(b_start)

  # Create CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                  fill_missing=True, overwrite=False).path
  src_mask_cv = None
  tgt_mask_cv = None

  broadcasting_field = cm.create(join(args.dst_path, 'field', 
                                      'stitch', 'broadcasting'),
                                 data_type='int16', num_channels=2,
                                 fill_missing=True, overwrite=False).path

  block_field = cm.create(join(args.dst_path, 'field', 'vvote'),
                          data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False).path

  final_dst = cm.create(join(args.dst_path, 'image_stitch{}'.format(args.suffix)),
                        data_type='uint8', num_channels=1, fill_missing=True,
                        overwrite=True).path

  # Task scheduling functions
  def remote_upload_it(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

  global_z_start = args.z_start
  global_z_end = args.z_stop

  chunk_grid = a.get_chunk_grid(cm, bbox, mip, 0, 1000, pad)

#  start_z = block_starts[0]+1
#  be = block_stops[0]
#  end_z = be+1 if be<global_z_end else be
#  influence_block = influencing_blocks_lookup[start_z]
#  t = a.stitch_compose_render_task(chunk_grid[0], src, final_dst, start_z,
#                                   end_z,
#                                             broadcasting_field, block_field, decay_dist,
#                                             influence_block,
#                                             mip, mip, pad, pad, chunk_size)
#

#  remote_upload(args.queue_name, t)

   # Task Scheduling Iterators
  class StitchComposeRenderT(object):
      def __init__(self, bs_list, be_list):
          self.bs_list = bs_list
          self.be_list = be_list
          print("bs list is ", self.bs_list)
          print("be list is ", self.be_list)

      def __iter__(self):
          for bs, be in zip(self.bs_list, self.be_list):
              start_z = bs+1
              influence_block = influencing_blocks_lookup[start_z]
              end_z = be+1 if be<global_z_end else be
              print("influence_block is", influence_block)
              t = a.stitch_compose_render_task(chunk_grid[0], src, final_dst, start_z, end_z,
                                               broadcasting_field, block_field, decay_dist,
                                               influence_block,
                                               mip, mip, pad, pad, chunk_size)
              yield from t

  ptask = []
  bs_list = make_range(block_starts, a.threads)
  be_list = make_range(block_stops, a.threads)
  print("bs_list is ", bs_list)
  print("be_list is ", be_list)
  #ptask.append(StitchComposeRenderT(block_starts, block_stops))
  #remote_upload_it(StitchComposeRenderT(block_starts, block_stops))
  for bs, be in zip(bs_list, be_list):
      ptask.append(StitchComposeRenderT(bs, be))
  #for i in ptask[0]:
  #    print(i)
  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload_it, ptask)

  start = time()
  #print("start until now time", start - begin_time)
  #a.wait_for_queue_empty(dst.path, 'load_image_done/{}'.format(mip), len(batch))
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Loading Tasks use time:", diff)

