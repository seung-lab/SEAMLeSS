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
    type=int, default=1024)
  parser.add_argument('--z_offset', type=int, default=-1,
    help='int for offset of section to be compared against')
  parser.add_argument('--block_size', type=int, default=10)
  parser.add_argument('--tgt_radius', type=int, default=3,
    help='int for number of sections to include in vector voting')
  parser.add_argument('--unnormalized', action='store_true', 
    help='do not normalize the CPC output, save as float')
  args = parse_args(parser)
  # Only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  max_mip = args.max_mip
  pad = args.pad

  # Compile ranges
  block_range = range(args.bbox_start[2], args.bbox_stop[2], args.block_size)
  overlap = args.tgt_radius
  full_range = range(args.block_size + overlap)
  # Create CloudVolume Manager
  src_info_path = join(args.src_path, 'image_blocks', 'even')
  cm = CloudManager(src_info_path, max_mip, pad, provenance)
  block_types = ['even', 'odd']
  # Create src CloudVolumes
  srcs = {}
  for block_type in block_types:
    src = cm.create(join(args.src_path, 'image_blocks', block_type), 
                    data_type='uint8', num_channels=1, fill_missing=True, 
                    overwrite=False)
    srcs[block_type] = src 

  data_type = 'uint8'
  if args.unnormalized:
    data_type = 'float32'
  
  # Create dst CloudVolumes for each block, since blocks will overlap by 3 sections
  dsts = {}
  for block_type in block_types:
    dst = cm.create(join(args.src_path, 'image_blocks', block_type, 'cpc',
                         '{}_{}'.format(args.src_mip, args.dst_mip),
                         '{}'.format(args.z_offset)),
                    data_type=data_type, num_channels=1, fill_missing=True, 
                    overwrite=True)
    dsts[block_type] = dst

  ##############
  # CPC script #
  ##############
  
  # Copy first section
  prefix = ''
  for i, block_start in enumerate(block_range):
    print('Scheduling CPC for block_start {}, block {} / {}'.format(block_start, i, 
                                                                    len(block_range)))
    batch = []
    for block_offset in full_range:
      block_type = block_types[i % 2]
      src = srcs[block_type]
      dst = dsts[block_type]
      z = block_start + block_offset
      t = a.cpc(cm, src, src, dst, z, z+args.z_offset, bbox, 
                    args.src_mip, args.dst_mip, norm=not args.unnormalized, prefix=prefix)
      batch.extend(t)
    run(a, batch)

  print('Check the task queue to note when finished')
