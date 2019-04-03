import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue

import sys
import torch
import json
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join, basename
from cloudmanager import CloudManager
from cloudvolume import Storage
from time import time
import tasks
import json
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

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src_path', type=str)
  parser.add_argument('--dst_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  args = parse_args(parser)
  dst_path = args.dst_path
  if not dst_path:
    dst_path = join(args.src_path, 'summary')
  args.max_mip = args.mip
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  
  # Simplify var names
  mip = args.mip
  pad = 0
  print('mip {}'.format(mip))

  # Compile ranges
  full_range = range(args.bbox_start[2], args.bbox_stop[2])
  # Create CloudVolume Manager
  cm = CloudManager(args.src_path, mip, pad, provenance)

  # Create src CloudVolumes
  src = cm.create(args.src_path, data_type='uint8', num_channels=1,
                     fill_missing=True, overwrite=False).path

  def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
        tq.insert_all(tasks)

  class SummarizeIterator():
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          for z in self.brange:
            t = [tasks.SummarizeTask(src, dst_path, z, bbox, mip)]
            yield from t

  range_list = make_range(full_range, a.threads)

  start = time()
  ptask = []
  for i in range_list:
      ptask.append(SummarizeIterator(i))
  if a.distributed:
    with ProcessPoolExecutor(max_workers=a.threads) as executor:
        executor.map(remote_upload, ptask)
  else:
      for t in ptask:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(t, args= [a])

  end = time()
  diff = end - start
  print("Sending SummarizeTask use time:", diff)
  start = time()
  print('Running Tasks')
  if a.distributed:
    a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("runtime:", diff)

  # compile summary files into one csv
  file_list = ['{}/{}'.format(bbox.stringify(0), z) for z in full_range]
  files = Storage(dst_path).get_files(file_list)
  s = 'z,error,'
  assert(len(files) > 0)
  i = 0
  header_file = files[i]['content']
  while not header_file:
    i += 1
    assert(i < len(files))
    header_file = files[i]['content']
  header_dict = json.loads(header_file.decode('utf-8'))
  header = []
  for i, (k, v) in enumerate(header_dict.items()):
    s += '{}'.format(k)
    header.append(k)
    if i < len(header_dict) - 1:
      s += ','
  s += '\n'
  for f in files:
    s += '{},'.format(basename(f['filename'])) 
    fd = json.loads(f['content'].decode('utf-8'))
    s += '{},'.format(fd is None)
    for i, k in enumerate(header):
      s += '{}'.format(fd[k])
      if i < len(header) - 1: 
        s += ','
    s += '\n'

  with Storage(dst_path) as stor:
    path = '{}.csv'.format(bbox.stringify(z_start=args.bbox_start[2], 
                                          z_stop=args.bbox_stop[2]))
    stor.put_file(path, s,
                  cache_control='no-cache')
    print('Save summary at {}'.format(join(dst_path, path)))
