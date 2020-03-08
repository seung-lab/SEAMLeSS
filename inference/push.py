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
from tasks import run
from boundingbox import BoundingBox
from aligner import Aligner
import numpy as np

def push_chunks_from_file(src_path, dst_path, failed_path, index, cv, mip):
    
    a = Aligner()
    cm = CloudManager(cv, 8, 512, None, batch_size=1, size_chunk=512, batch_mip=mip, create_info=False)
    cv = cm.create(cv, data_type='int16', num_channels=2, fill_missing=False, overwrite=False)
    chunks = np.loadtxt('{path}/{i}'.format(path = src_path, i = index), dtype='int64')
    pivots = np.loadtxt('{path}/{i}_chunks'.format(path = src_path, i = index), dtype='int64')
    pivots = pivots - index
    chunks_pushed = a.push_coordinate_chunks(cv, mip, chunks, pivots)
    np.savetxt('{path}/{i}'.format(path = dst_path, i = index), chunks_pushed, '%.0f')
    np.savetxt('{path}/{i}'.format(path = failed_path, i = index), chunks_failed, '%.0f')


def get_pivots(array, ran):
    pivots = list()
    ri = 0
    top = ran[ri+1]
    start = array[0]
    while start >= top:
        ri = ri + 1
        top = ran[ri+1]
    for i in range(0, len(array)):
        if array[i] >= top:
            pivots.append(i)
        while array[i] >= top:
            ri = ri + 1
            top = ran[ri+1]
    return pivots
        

def chunk_nparray(array, cv, mip, xyz=[1,2,3], scale=[4,4,40]):
    if not mip in cv.cvs.keys():
        cv.create(mip)
    c = cv.cvs[mip]
    x_loc = xyz[0]
    y_loc = xyz[1]
    z_loc = xyz[2]
    x_chunk = c.chunk_size[0] * (2**mip) * scale[0]
    y_chunk = c.chunk_size[1] * (2**mip) * scale[1]
    z_chunk = c.chunk_size[2] * scale[2]
    bb = c.mip_bounds(0)
    x_range = range(bb.minpt[0] * scale[0], bb.maxpt[0] * scale[0], x_chunk)
    y_range = range(bb.minpt[1] * scale[1], bb.maxpt[1] * scale[1], y_chunk)
    z_range = range(bb.minpt[2] * scale[2], bb.maxpt[2] * scale[2], z_chunk)
    sorted_z = array[np.argsort(array[:, z_loc])]
    z_pivots = np.argwhere(sorted_z[:,z_loc][0:-1] != sorted_z[:,z_loc][1:]).flatten() + 1
    z_arrays = np.split(sorted_z, z_pivots)
    l = []
    for z_array in z_arrays:
        sorted_y = z_array[np.argsort(z_array[:, y_loc])]
        sorted_y_view = sorted_y[:, y_loc]
        #print("y: " + str(sorted_y_view.shape))
        #if sorted_y_view.shape[0] == 5:
        # return sorted_y
        pivots_y = get_pivots(sorted_y_view, y_range)
        y_arrays = np.split(sorted_y, pivots_y)
        for y_array in y_arrays:
         sorted_x = y_array[np.argsort(y_array[:, x_loc])]
         sorted_x_view = sorted_x[:, x_loc]
         #print("x: " + str(sorted_x_view.shape))
         pivots_x = get_pivots(sorted_x_view, x_range)
         x_arrays = np.split(sorted_x, pivots_x)
         l.extend(x_arrays)
    pivots = []
    cumulative = 0
    for arr in l:
        cumulative = cumulative + len(arr)
        pivots.append(cumulative)
    return np.vstack(l), pivots

def save_chunked_array(arr, pivots, num_chunks, path):
  pivot_chunks = np.array_split(np.array(pivots), num_chunks)
  start = 0
  for i in range(0, num_chunks):
    end = pivot_chunks[i][-1]
    np.savetxt('{path}/{i}'.format(path = path, i = start), arr[start:end], '%.0f')
    np.savetxt('{path}/{i}_chunks'.format(path = path, i = start), pivot_chunks[i], '%.0f')
    start = end

	

def make_range(block_range, part_num):
    rangelen = len(block_range)
    if(rangelen < part_num):
        srange = 1
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
  parser.add_argument('--section_lookup', type=str, 
    help='path to json file with section specific settings')
  parser.add_argument('--z_range_path', type=str, 
    help='path to csv file with list of z indices to use')
  parser.add_argument('--field_path', type=str)
  parser.add_argument('--info_path', type=str,
    help='path to CloudVolume to use as template info file')
  parser.add_argument('--field_mip', type=int)
  parser.add_argument('--field_mip', type=int)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--failed_queue', type=str, default="",
    help='failed queue to keep track of tasks that do not fit into GPU')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--use_cpu',
     help='use CPU as torch.device',
     action='store_true')
  args = parse_args(parser)
  # only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  chunk_size = 512

  src_mip = args.src_mip
  max_mip = args.max_mip
  pad = args.pad

  # Compile ranges
  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  if args.z_range_path:
    print('Compiling z_range from {}'.format(args.z_range_path))
    z_range = []
    with open(args.z_range_path) as f:
      reader = csv.reader(f, delimiter=',')
      for k, r in enumerate(reader):
         if k != 0:
           z_start = int(r[0])
           z_stop  = int(r[1])
           print('adding to z_range {}:{}'.format(z_start, z_stop))
           z_range.extend(list(range(z_start, z_stop)))

  # Create CloudVolume Manager
  if args.info_path:
    template_path = args.info_path
    cm = CloudManager(template_path, max_mip, pad, provenance, batch_size=1,
                      size_chunk=chunk_size, batch_mip=src_mip, 
                      create_info=False)
  else:
    template_path = args.src_path
    cm = CloudManager(template_path, max_mip, pad, provenance, batch_size=1,
                      size_chunk=chunk_size, batch_mip=src_mip, 
                      create_info=True)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='int16', num_channels=2,
                     fill_missing=True, overwrite=False)
  dst = cm.create(args.dst_path, data_type='int16', num_channels=2,
                     fill_missing=True, overwrite=True)

  # Source Dict
  src_path_to_cv = {args.src_path: src}

  # compile model lookup per z index
  affine_lookup = None
  source_lookup = {}
  if args.section_lookup:
    affine_lookup = {}
    with open(args.section_lookup) as f:
      section_list = json.load(f)
      for section in section_list:
        z = section['z']
        affine_lookup[z] = np.array(section['transform'])
        affine_lookup[z][:, 2] += args.downsample_shift

        try:
          src_path = section['src']
        except KeyError:
          src_path = args.src_path

        if src_path not in src_path_to_cv:
          src_path_to_cv[src_path] = cm.create(src_path,
              data_type='uint8', num_channels=1, fill_missing=True,
              overwrite=False)
        source_lookup[z] = src_path_to_cv[src_path]

  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

  class InvertTaskIterator(object):
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

          try:
            src_path = source_lookup[z].path
            if src_path != src.path:
              print("Overriding {} source dir with path {}".format(z, src_path))
          except KeyError:
            src_path = src.path
          
          t = a.invert_get_tasks_batch(cm, src_path, dst.path, z, bbox,
                           src_mip, pad, failed_queue=args.failed_queue, use_cpu=args.use_cpu) 
          yield from t

  ptask = []
  range_list = make_range(z_range, a.threads)
  start = time()
  for irange in range_list:
      ptask.append(InvertTaskIterator(irange))

  if a.distributed:
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
  else:
    for t in ptask:
     tq = LocalTaskQueue(parallel=1)
     tq.insert_all(t, args=[a])

  end = time()
  diff = end - start
  print("Sending Invert Tasks use time:", diff)
  print('Running Invert Tasks')
  # wait 
  start = time()
  # a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Invert Tasks use time:", diff)

