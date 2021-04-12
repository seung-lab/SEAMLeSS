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
import numpy as np
from cloudvolume import CloudVolume

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
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--info_path', type=str,
    help='path to CloudVolume to use as template info file')
  parser.add_argument('--field_path', type=str)
  parser.add_argument('--field_mip', type=int)
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
    type=int, default=512)
  parser.add_argument('--render_pad',
    help='the size of the largest displacement expected; should be 2^high_mip',
    type=int, default=512)
  args = parse_args(parser)
  # only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  # chunk_size = 1024
  chunk_size = 1024


  src_mip = args.src_mip
  field_mip = args.field_mip
  max_mip = args.max_mip
  pad = args.pad
  render_pad = args.render_pad

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
  src_cv = CloudVolume(args.src_path)
  field_cv = CloudVolume(args.field_path)
  src = cm.create(args.src_path, data_type=str(src_cv.dtype), num_channels=1,
                     fill_missing=True, overwrite=False)
  field = cm.create(args.field_path, data_type=str(field_cv.dtype), num_channels=2,
                         fill_missing=True, overwrite=False)
  dst = cm.create(args.dst_path, data_type=str(src_cv.dtype), num_channels=1,
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
              data_type=str(src_cv.dtype), num_channels=1, fill_missing=True,
              overwrite=False)
        source_lookup[z] = src_path_to_cv[src_path]

  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

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

          try:
            src_path = source_lookup[z].path
            if src_path != src.path:
              print("Overriding {} source dir with path {}".format(z, src_path))
          except KeyError:
            src_path = src.path

          t = a.render(cm, src_path, field.path, dst.path, z, z, z, bbox,
                           src_mip, field_mip, affine=affine, pad=render_pad)
          yield from t

  ptask = []
  range_list = make_range(z_range, a.threads)
  start = time()
  for irange in range_list:
      ptask.append(RenderTaskIterator(irange))

  if a.distributed:
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
  else:
    for t in ptask:
     tq = LocalTaskQueue(parallel=1)
     tq.insert_all(t, args=[a])

  end = time()
  diff = end - start
  print("Sending Render Tasks use time:", diff)
  print('Running Render Tasks')
  # wait
  start = time()
  # a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Render Tasks use time:", diff)

