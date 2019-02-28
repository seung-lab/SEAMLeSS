import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from time import time
from tasks import run 

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--model_path', type=str,
    help='relative path to the ModelArchive to use for computing fields')
  parser.add_argument('--src_path1', type=str)
  parser.add_argument('--src_path2', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
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
  parser.add_argument('--block_size', type=int, default=10)
  args = parse_args(parser)
  # Only compute matches to previous sections
  args.serial_operation = True
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  mip = args.mip
  max_mip = args.max_mip
  pad = args.max_displacement

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_path1, max_mip, pad, provenance, batch_size=1,
                    size_chunk=128, batch_mip=mip) 
  #cm = CloudManager(args.src_path1, max_mip, pad, provenance)

  # Create src CloudVolumes
  src1 = cm.create(args.src_path1, data_type='float', num_channels=1,
                     fill_missing=True, overwrite=False)

  src2 = cm.create(args.src_path2, data_type='float', num_channels=1,
                     fill_missing=True, overwrite=False)

  # Create dst CloudVolumes
  dst = cm.create(join(args.dst_path, 'image'),
                  data_type='float32', num_channels=1, fill_missing=True,
                  overwrite=True)
  batch =[]
  prefix = str(mip)
  for z in full_range:
      #print("Fcorr for z={} and z={}".format(z, z+1))
      t = a.mask_op(cm, bbox, mip, z, z, src1, src2, dst, z)
      batch.extend(t)
  start = time()
  run(a, batch)
  end = time()
  diff = end - start
  print_run(diff, len(batch))
  start = time()
  # wait 
  n = len(batch)
  a.wait_for_queue_empty(dst.path, 'Mask_op_done/{}'.format(prefix), n)
  end = time()
  diff = end - start
  print_run(diff, len(batch))

