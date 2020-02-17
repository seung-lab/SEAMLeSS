import gevent.monkey
gevent.monkey.patch_all()

from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue, MockTaskQueue

import argparse
from args import get_argparser, \
                 parse_args, \
                 get_aligner, \
                 get_bbox, \
                 get_provenance
from os.path import join
from temporal_regularization import create_field_bump
from time import time, sleep
import json
from cloudmanager import CloudManager

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
    parser = get_argparser(description='Convolve field with Gaussian')
    parser.add_argument('--src_paths',
        type=json.loads,
        help='json dict of z_offsets to CloudVolume paths of input fields')
    parser.add_argument('--dst_path',
        type=str,
        help='CloudVolume path where convolved field will be written')
    parser.add_argument('--mip',
        type=int,
        help='int for MIP level of SRC_PATH')
    parser.add_argument('--z_start',
        type=int,
        help='int for first section in range to be aligned')
    parser.add_argument('--z_stop',
        type=int,
        help='int for last section in range to be aligned')
    parser.add_argument('--max_mip',
        type=int,
        default=9,
        help='int for largest MIP level for chunk alignment')
    parser.add_argument('--bbox_start',
        nargs=3,
        type=int,
        help='bbox origin, 3-element int list')
    parser.add_argument('--bbox_stop',
        nargs=3,
        type=int,
        help='bbox origin+shape, 3-element int list')
    parser.add_argument('--pad',
        help='int for size of max displacement expected; should be 2^max_mip',
        type=int, default=2048)
    parser.add_argument('--sigma',
        type=float,
        default=1.4,
        help='float for std of Gaussian kernel')
    args = parse_args(parser)
    # Only compute matches to previous sections
    a = get_aligner(args)
    args.bbox_mip = 0
    bbox = get_bbox(args)
    provenance = get_provenance(args)
    chunk_size = 1024

    # Create CloudVolume Manager
    cm = CloudManager(cv_path=list(args.src_paths.values())[0],
                      max_mip=args.max_mip,
                      max_displacement=args.pad,
                      provenance=provenance,
                      batch_size=1,
                      size_chunk=chunk_size,
                      batch_mip=args.mip)

    # Create field CloudVolumes
    print('Create src field CloudVolumes')
    src_fields = {}
    for k, src_path in args.src_paths.items():
        src_fields[int(k)] = cm.create(path=src_path,
                                  data_type='int16',
                                  num_channels=2,
                                  fill_missing=True,
                                  overwrite=False).path

    print('Create dst field CloudVolume')
    dst = cm.create(path=args.dst_path,
                    data_type='int16',
                    num_channels=2,
                    fill_missing=True,
                    overwrite=True).path

    z_range = range(args.bbox_start[2], args.bbox_stop[2])
    offset_range = [int(k) for k in args.src_paths.keys()]

    # Task scheduling functions
    def remote_upload(tasks):
        with GreenTaskQueue(queue_name=args.queue_name) as tq:
            tq.insert_all(tasks)  

    def execute(task_iterator, z_range):
        if len(z_range) > 0:
            ptask = []
            range_list = make_range(z_range, a.threads)
            start = time()

        for irange in range_list:
            ptask.append(task_iterator(irange))
        if args.dry_run:
            for t in ptask:
                tq = MockTaskQueue(parallel=1)
                tq.insert_all(t, args=[a])
        else:
            if a.distributed:
                with ProcessPoolExecutor(max_workers=a.threads) as executor:
                    executor.map(remote_upload, ptask)
            else:
                for t in ptask:
                    tq = LocalTaskQueue(parallel=1)
                    tq.insert_all(t, args=[a])
 
        end = time()
        diff = end - start
        print('Sending {} use time: {}'.format(task_iterator, diff))
        if a.distributed:
            print('Run {}'.format(task_iterator))
            # wait
            start = time()
            a.wait_for_sqs_empty()
            end = time()
            diff = end - start
            print('Executing {} use time: {}\n'.format(task_iterator, diff))

    class GaussianFilterZ(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                # fields are stored at src_z
                z_list = [z for z_offset in offset_range]
                t = a.gaussian_filter_z(cm=cm,
                                        src_cv_list=list(src_fields.values()),
                                        dst_cv=dst,
                                        z_list=z_list,
                                        dst_z=z,
                                        bbox=bbox,
                                        mip=args.mip,
                                        sigma=args.sigma)
                yield from t

    # Gaussian filter fields
    print('CREATE PAIRWISE FIELDS')
    execute(GaussianFilterZ, z_range)
