import sys
import torch
import json
import math
import csv
from time import time, sleep
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
import numpy as np
from cloudmanager import CloudManager
from tasks import run 

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--affine_lookup', type=str, 
    help='path to csv of affine transforms indexed by section')
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--field_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--src_mip', type=int)
  parser.add_argument('--field_mip', type=int)
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
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--use_sqs_wait', action='store_true',
    help='wait for SQS to return that its queue is empty; incurs fixed 30s for initial wait')
  args = parse_args(parser)
  # Only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  chunk_size = 1024
  
  # Simplify var names
  src_mip = args.src_mip
  field_mip = args.field_mip
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
  field = cm.create(args.field_path, data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)
  dst = cm.create(args.dst_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=True)

  # compile model lookup per z index
  affine_lookup = None
  if args.affine_lookup:
    affine_lookup = {}
    with open(args.affine_lookup) as f:
      reader = csv.reader(f, delimiter=',')
      for k, r in enumerate(reader):
        if k != 0:
          a = float(r[0])
          b = float(r[1])
          c = float(r[2])
          d = float(r[3])
          e = float(r[4])
          f = float(r[5])
          z = float(r[6])
          affine = np.array([[a,b,c],[d,e,f]])
          affine_lookup[z] = affine

  # Render sections
  batch = []
  prefix = ''
  for z in z_range:
    affine = None
    if affine_lookup:
      affine = affine_lookup[z]
    t = a.render(cm, src, field, dst, z, z, z, bbox, src_mip, field_mip, 
                   affine=affine, prefix=prefix)
    batch.extend(t)
  print('Scheduling RenderTasks')
  start = time()
  run(a, batch)
  end = time()
  diff = end - start
  print_run(diff, len(batch))
