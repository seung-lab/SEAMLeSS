import gevent.monkey

gevent.monkey.patch_all()

import csv
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from os.path import join
from time import time, sleep
from math import floor

from taskqueue import GreenTaskQueue, LocalTaskQueue, MockTaskQueue, TaskQueue

from args import get_aligner, get_argparser, get_provenance, parse_args
from boundingbox import BoundingBox
from cloudmanager import CloudManager
import numpy as np

import json
from mask import Mask

def make_range(block_range, part_num):
    rangelen = len(block_range)
    if rangelen < part_num:
        srange = 1
        part = rangelen
    else:
        part = part_num
        srange = rangelen // part
    range_list = []
    for i in range(part - 1):
        range_list.append(block_range[i * srange : (i + 1) * srange])
    range_list.append(block_range[(part - 1) * srange :])
    return range_list

if __name__ == "__main__":
    parser = get_argparser()
    parser.add_argument('--src_mask', action='append',
            help='Pass string that contains a JSON dict. Fields: "cv", "mip", "val", "op"',
            type=json.loads, dest='src_masks')
    parser.add_argument("--dst_path", type=str)
    parser.add_argument("--z_start", type=int)
    parser.add_argument("--z_stop", type=int)
    parser.add_argument("--mip", type=int)
    parser.add_argument("--max_mip", type=int, default=9)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--pad", type=int, default=1024)
    parser.add_argument(
        "--field_path",
        type=str,
        help="if specified, applies field to source before aligning to target",
    )
    parser.add_argument(
        "--blackout_op",
        type=str,
        default='none'
    )
    parser.add_argument('--info_path', type=str, help='path to CloudVolume to use as template info file')
    args = parse_args(parser)
    a = get_aligner(args)
    chunk_size = args.chunk_size
    def remote_upload(tasks):
        with GreenTaskQueue(queue_name=args.queue_name) as tq:
            tq.insert_all(tasks)

    max_mip = args.max_mip
    mip = args.mip
    bbox = BoundingBox(0, 491520, 0, 491520, 0, args.max_mip)
    src_masks = []
    src_masks = [Mask(**m) for m in args.src_masks]
    provenance = get_provenance(args)
    cm = CloudManager(args.info_path, max_mip, args.pad, provenance, batch_size=1,
                    size_chunk=chunk_size, batch_mip=args.mip)
    dst = cm.create(args.dst_path, data_type='uint8', num_channels=1, fill_missing=True, overwrite=True).path

    field = cm.create(
        args.field_path,
        data_type='float32',
        num_channels=2,
        fill_missing=True,
        overwrite=False,
    ).path
    
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
            print("Sending {} use time: {}".format(task_iterator, diff))
            if a.distributed:
                print("Run {}".format(task_iterator))
                # wait
                start = time()
                a.wait_for_sqs_empty()
                end = time()
                diff = end - start
                print("Executing {} use time: {}\n".format(task_iterator, diff))
    
    class StitchFinalRender(object):
        def __init__(self, z_range):
          self.z_range = z_range

        def __iter__(self):
          for z in self.z_range:
            t = a.render_masks(cm, dst, field, src_z=z,
                         field_z=z, dst_z=z, bbox=bbox,
                         dst_mip=mip, pad=args.pad,
                         masks=src_masks, blackout_op=args.blackout_op)
            yield from t

    compose_range = range(args.z_start, args.z_stop)
    execute(StitchFinalRender, compose_range)
