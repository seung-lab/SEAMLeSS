import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue

import sys
import torch
import json
import numpy as np
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from time import time
from tasks import run 

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


def find_section_clamping_values(zlevel, lowerfract, upperfract):
    """compute the clamping values for each section."""
    filtered = np.copy(zlevel)

    # remove pure black from frequency counts as
    # it has no information in our images
    filtered[0] = 0

    cdf = np.zeros(shape=(len(filtered), ), dtype=np.uint64)
    cdf[0] = filtered[0]
    for i in range(1, len(filtered)):
        cdf[i] = cdf[i - 1] + filtered[i]

    total = cdf[-1]

    if total == 0:
        return (0, 0)

    lower = 0
    for i, val in enumerate(cdf):
        if float(val) / float(total) > lowerfract:
            break
        lower = i

    upper = 0
    for i, val in enumerate(cdf):
        if float(val) / float(total) > upperfract:
            break
        upper = i

    return (lower, upper)

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--z_offset', type=int,
    help='distance between src_z and tgt_z')
  parser.add_argument('--src_mip', type=int)
  parser.add_argument('--dst_mip', type=int)
  parser.add_argument('--fill_value', type=int, default=0)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--pad', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--preprocessor_path', type=str, default='',
    help='Optional preprocessor to apply to image cutouts')
  parser.add_argument('--luminance_level_path', type=str,
    help='Optional luminance levels per section, used for per section contrast adjustment')
  # parser.add_argument('--save_intermediary', action='store_true')
  args = parse_args(parser)
  args.max_mip = args.dst_mip
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  src_mip = args.src_mip
  dst_mip = args.dst_mip
  fcorr_chunk_size = 2**(dst_mip-src_mip)
  chunk_size = 128 
  a.chunk_size = (chunk_size, chunk_size)
  max_mip = args.max_mip
  pad = args.pad
  z_offset = args.z_offset
  fill_value = args.fill_value
  preprocessor_path = args.preprocessor_path
  luminance_level_path = args.luminance_level_path

  print('src_mip {}'.format(src_mip))
  print('dst_mip {}'.format(dst_mip))
  print('fcorr_chunk_size {}'.format(fcorr_chunk_size))
  # print('chunk_size {}'.format(chunk_size))
  print('z_offset {}'.format(z_offset))
  print('preprocessor: {}'.format(preprocessor_path or "None"))
  print('luminance_level_path: {}'.format(luminance_level_path or "None"))

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=dst_mip)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False)

  fcorr_dir = 'fcorr/{}_{}/{}'.format(src_mip, dst_mip, z_offset)
  # Create dst CloudVolumes
  dst_post = cm.create(join(args.dst_path, fcorr_dir, 'post'),
                  data_type='float32', num_channels=1, fill_missing=True,
                  overwrite=True)
  dst_pre = cm.create(join(args.dst_path, fcorr_dir, 'pre'),
                  data_type='float32', num_channels=1, fill_missing=True,
                  overwrite=True)

  # Load luminance levels + compute cutoffs
  if luminance_level_path is not None:
    with open(luminance_level_path, "r") as f:
      luminance_level = json.load(f)
  else:
    luminance_level = None

  prefix = str(src_mip)
  class TaskIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
            #print("Fcorr for z={} and z={}".format(z, z+1))
            if luminance_level is not None:
              lower_cut, upper_cut = find_section_clamping_values(luminance_level[str(z)], 0.0023, 1.0-0.01)
            else:
              lower_cut, upper_cut = 0, 255

            min_val, max_val = 1, 255
            t = a.compute_fcorr(cm, src.path, dst_pre.path, dst_post.path, bbox, 
                                src_mip, dst_mip, z, z+args.z_offset, z, 
                                fcorr_chunk_size, fill_value=fill_value,
                                preprocessor_path=preprocessor_path,
                                lower_bound=(lower_cut/255.0, min_val/255.0),
                                upper_bound=(upper_cut/255.0, max_val/255.0)
            )
            yield from t

  range_list = make_range(full_range, a.threads)

  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)

  start = time()
  ptask = []
  for i in range_list:
      ptask.append(TaskIterator(i))

  if a.distributed:
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
  else:
      for t in ptask:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(t, args= [a])


  end = time()
  diff = end - start
  print("Sending Tasks use time:", diff)
  print('Running Tasks')
  # wait
