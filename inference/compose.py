import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
from taskqueue import GreenTaskQueue

import csv
from time import time
from args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from cloudmanager import CloudManager

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
  parser.add_argument('--z_range_path', type=str, 
    help='path to csv file with list of z indices to use')
  parser.add_argument('--info_path', type=str,
    help='path to CloudVolume to use as template info file')
  parser.add_argument('--field_path', type=str)
  parser.add_argument('--fine_field_path', type=str)
  parser.add_argument('--coarse_field_path', type=str)
  parser.add_argument('--fine_mip', type=int)
  parser.add_argument('--coarse_mip', type=int)
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
  args = parse_args(parser)
  # only compute matches to previous sections
  a = get_aligner(args)
  bbox = get_bbox(args)
  provenance = get_provenance(args)
  chunk_size = 1024

  fine_mip = args.fine_mip
  coarse_mip = args.coarse_mip
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
  template_path = args.src_path
  if args.info_path:
    template_path = args.info_path
  cm = CloudManager(template_path, max_mip, pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=fine_mip)

  # Create src CloudVolumes
  fine_field = cm.create(args.fine_field_path, data_type='int16', num_channels=2,
                         fill_missing=True, overwrite=False)
  coarse_field = cm.create(args.coarse_field_path, data_type='int16', num_channels=2,
                          fill_missing=True, overwrite=False)
  field = cm.create(args.field_path, data_type='int16', num_channels=2,
                    fill_missing=True, overwrite=True)

  def remote_upload(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)


  class ComposeTaskIterator(object):
      def __init__(self, zrange):
          self.zrange = zrange
      def __iter__(self):
          print("range is ", self.zrange)
          for z in self.zrange:
              t = a.compose(cm, fine_field.path, coarse_field.path,
                            field.path, z, z, z, bbox, fine_mip,
                            coarse_mip, fine_mip, factor=1, affine=None,
                            pad=pad)
              yield from t

  ptask = []
  range_list = make_range(z_range, a.threads)
  
  start = time()
  for irange in range_list:
      ptask.append(ComposeTaskIterator(irange))
  
  with ProcessPoolExecutor(max_workers=a.threads) as executor:
      executor.map(remote_upload, ptask)
 
  end = time()
  diff = end - start
  print("Sending Compose Tasks use time:", diff)
  print('Running Compose Tasks')
  # wait 
  start = time()
  a.wait_for_sqs_empty()
  end = time()
  diff = end - start
  print("Executing Compose Tasks use time:", diff)
