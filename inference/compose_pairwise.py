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
from time import time, sleep
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
    parser = get_argparser()
    parser.add_argument('--src_path',
        type=str,
        help='CloudVolume path for field to use in composition')
    parser.add_argument('--dst_path',
        type=str,
        help='CloudVolume path where composed field will be written')
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
    args = parse_args(parser)
    # Only compute matches to previous sections
    a = get_aligner(args)
    args.bbox_mip = 0
    bbox = get_bbox(args)
    provenance = get_provenance(args)
    chunk_size = 1024

    # Create CloudVolume Manager
    cm = CloudManager(cv_path=args.src_path,
                      max_mip=args.max_mip,
                      max_displacement=args.pad,
                      provenance=provenance,
                      batch_size=1,
                      size_chunk=chunk_size,
                      batch_mip=args.mip)

    z_range = range(args.z_start, args.z_stop+1)

    src = cm.create(path=args.src_path,
                          data_type='int16',
                          num_channels=2,
                          fill_missing=True,
                          overwrite=False).path
    dst = cm.create(path=args.dst_path,
                          data_type='int16',
                          num_channels=2,
                          fill_missing=True,
                          overwrite=True).path

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
    
    class ComposePairwiseFields(object):
        def __init__(self, zrange):
            self.zrange = zrange
        def __iter__(self):
            print("range is ", self.zrange)
            for z in self.zrange:
                affine = None
                t = a.compose(cm=cm, 
                              f_cv=dst, 
                              g_cv=src,
                              dst_cv=dst,
                              f_z=z-1, 
                              g_z=z, 
                              dst_z=z, 
                              bbox=bbox, 
                              f_mip=args.mip,
                              g_mip=args.mip, 
                              dst_mip=args.mip, 
                              factor=1, 
                              affine=None,
                              pad=args.pad)
                yield from t

    # Pairwise alignment
    print('COMPOSE PAIRWISE FIELDS')
    for z in z_range:
        execute(ComposePairwiseFields, [z])
