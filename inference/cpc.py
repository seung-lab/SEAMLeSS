import sys
import torch
import json
from time import time, sleep
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from tasks import run 

def print_run(diff, n_tasks):
  print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--src_info_path', type=str, default='',
    help='str to existing CloudVolume path to use as template for new CloudVolumes')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--src_mip', type=int,
    help='int for input MIP')
  parser.add_argument('--dst_mip', type=int,
    help='int for output MIP, which will dictate the size of the block used')
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--z_offset', type=int, default=-1,
    help='int for offset of section to be compared against')
  parser.add_argument('--unnormalized', action='store_true', 
    help='do not normalize the CPC output, save as float')
  args = parse_args(parser)
  if args.src_info_path == '':
    args.src_info_path = args.src_path
  # Only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  max_mip = args.max_mip
  pad = args.pad

  # Compile ranges
  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_info_path, max_mip, pad, provenance)
  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)
  data_type = 'uint8'
  if args.unnormalized:
    data_type = 'float32'
  
  # Create dst CloudVolumes for each block, since blocks will overlap by 3 sections
  dst = cm.create(join(args.src_path, 'cpc', '{}_{}'.format(args.src_mip, args.dst_mip),
                       '{}'.format(args.z_offset)), 
                  data_type=data_type, num_channels=1, fill_missing=True, 
                  overwrite=True)

  ##############
  # CPC script #
  ##############
  k = 0
  batch = []
  prefix = ''
  for z in z_range:
    t = a.cpc(cm, src, src, dst, z, z+args.z_offset, bbox, 
                  args.src_mip, args.dst_mip, norm=not args.unnormalized, prefix=prefix)
    batch.extend(t)
    k += 1
    if k >= 100:
      print('Scheduling CPC for {} tasks'.format(len(batch)))
      run(a, batch)
      batch = []
      k = 0

  run(a, batch)
  print('Finished scheduling. Watch queue for finish.')

