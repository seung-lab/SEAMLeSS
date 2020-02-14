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
        help='path to CloudVolume image to be aligned')
    parser.add_argument('--dst_path',
        type=str,
        help='path to CloudVolume where aligned image will be written')
    parser.add_argument('--model_path',
        type=str,
        help='path to SEAMLeSS model repository')
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
    parser.add_argument('--radius',
        type=int,
        default=3,
        help='int for range of neighbors to use as targets')
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

    # Create src CloudVolumes
    print('Create src image CloudVolume')
    src = cm.create(path=args.src_path,
                    data_type='uint8',
                    num_channels=1,
                    fill_missing=True,
                    overwrite=False).path

    print('Create dst image CloudVolume')
    dst_img_path = join(args.dst_path, 'image')
    dst = cm.create(path=dst_img_path,
                    data_type='uint8',
                    num_channels=1,
                    fill_missing=True,
                    overwrite=True).path

    z_range = range(args.z_start, args.z_stop+1)
    offset_range = [i for i in range(-args.radius, args.radius+1)]

    # Create field CloudVolumes
    print('Creating field CloudVolumes')
    pair_fields = {}
    for k in offset_range:
        path = join(args.dst_path, 'field', 'pairs', str(k))
        pair_fields[k] = cm.create(path=path,
                                  data_type='int16',
                                  num_channels=2,
                                  fill_missing=True,
                                  overwrite=True).path
    reg_field_path = join(args.dst_path, 'field', 'regularized')
    reg_field = cm.create(path=reg_field_path,
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

    class ComputePairwiseFields(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                for z_offset in offset_range:
                    field = pair_fields[z_offset]
                    t = a.compute_field(cm=cm, 
                                model_path=args.model_path, 
                                src_cv=src, 
                                tgt_cv=src, 
                                field_cv=field, 
                                src_z=z, 
                                tgt_z=z+z_offset, 
                                bbox=bbox, 
                                mip=args.mip, 
                                pad=args.pad,
                                src_mask_cv=None,
                                src_mask_mip=0, 
                                src_mask_val=0,
                                tgt_mask_cv=None, 
                                tgt_mask_mip=0, 
                                tgt_mask_val=0, 
                                prev_field_cv=None, 
                                prev_field_z=None)
                    yield from t

    # Pairwise alignment
    print('START ALIGNMENT')
    print('CREATE PAIRWISE FIELDS')
    execute(ComputePairwiseFields, z_range)
    # print('CREATE CONFIDENCE MAPS')
    # execute(CreateConfidenceMaps, z_range)
    print('REGULARIZE FIELDS')
    # execute(RegularizeFields, z_range)