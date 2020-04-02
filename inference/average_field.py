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
from cloudvolume import CloudVolume
import tasks

import json
from mask import Mask

def print_run(diff, n_tasks):
    if n_tasks > 0:
        print(
            ": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks)
        )

def interpolate(x, start, stop_dist):
  """Return interpolation value of x for range(start, stop)

  Args
     x: int location
     start: location corresponding to 1
     stop_dist: distance from start corresponding to 0
  """
  assert(stop_dist != 0)
  stop = start + stop_dist
  d = (stop - x) / (stop - start)
  return min(max(d, 0.), 1.)



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


def ranges_overlap(a_pair, b_pair):
    a_start, a_stop = a_pair
    b_start, b_stop = b_pair
    return (
        (b_start <= a_start and b_stop >= a_start)
        or (b_start >= a_start and b_stop <= a_stop)
        or (b_start <= a_stop and b_stop >= a_stop)
    )


if __name__ == "__main__":
    parser = get_argparser()
    parser.add_argument(
        "--field_path",
        type=str,
        help="if specified, applies field to source before aligning to target",
    )
    parser.add_argument(
        "--mip",
        type=int,
        help="MIP level of the primer. E.g. the MIP of a coarse alignment",
    )
    parser.add_argument(
        "--max_mip",
        type=int,
        default=9
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--field_dtype",
        type=str,
        help="Data type of coarse vector field (typically int16 or float32)",
        default='float32'
    )
    parser.add_argument(
        "--output_field_path",
        type=str,
        help="if specified, applies field to source before aligning to target",
    )
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--z_start", type=int)
    parser.add_argument("--z_stop", type=int)
    parser.add_argument('--block_overlap', type=int, default=4)
    parser.add_argument('--generate_params_from_skip_file', type=str, default=None)
    parser.add_argument('--render_dst', type=str)
    parser.add_argument('--pad', type=int, default=512)
    parser.add_argument(
        "--skip_average",
        action='store_true',
        help="If True, skip compute field and vector voting"
    )
    # parser.add_argument("--chunk_size", type=int, default=1024)

    args = parse_args(parser)
    # Only compute matches to previous sections
    args.serial_operation = True
    a = get_aligner(args)
    provenance = get_provenance(args)
    chunk_size = args.chunk_size
    # Simplify var names
    mip = args.mip
    max_mip = args.max_mip
    block_size = args.block_size
    pad = args.pad

    field_path = args.field_path
    field_dtype = args.field_dtype
    output_field_path = args.output_field_path
    render_dst = args.render_dst
    do_average = not args.skip_average

    # Create CloudVolume Manager
    cm = CloudManager(
        args.field_path,
        max_mip,
        pad,
        provenance,
        batch_size=1,
        size_chunk=chunk_size,
        batch_mip=mip,
        create_info=False
    )

    # Create src CloudVolumes
    print("Create src & align image CloudVolumes")
    field = cm.create(
        field_path,
        data_type=field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=False,
    ).path

    high_freq_field = cm.create(
        join(field_path, 'high_freq'),
        data_type=field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=True,
    ).path

    # import ipdb
    # ipdb.set_trace()

    volume_size = list(np.array(cm.vec_total_sizes[mip][0:2]) // np.array(cm.dst_chunk_sizes[mip]))

    avg_field_path = join(output_field_path, "_avg_chunk")
    avg_field_info = CloudVolume.create_new_info(
        num_channels=2,
        layer_type='image',
        encoding='raw',
        data_type=field_dtype,
        resolution=cm.info['scales'][mip]['resolution'],
        voxel_offset=[0,0,0],
        chunk_size=[1, 1, 1],
        # volume_size=[*volume_size, 28000],
        volume_size=[*volume_size, 4500],
    )
    avg_field_vol = CloudVolume(avg_field_path, info=avg_field_info)
    avg_field_vol.commit_info()
    # bbox = BoundingBox(0, 491520, 0, 491520, 0, args.max_mip)
    bbox = BoundingBox(150000, 180000, 50000, 80000, 0, args.max_mip)
    avg_chunk_bbox = BoundingBox(0, volume_size[0], 0, volume_size[1], 0, 0)

    avg_field_section_path = join(output_field_path, "_avg_section")
    avg_field_section_info = CloudVolume.create_new_info(
        num_channels=2,
        layer_type='image',
        encoding='raw',
        data_type=field_dtype,
        resolution=cm.info['scales'][mip]['resolution'],
        voxel_offset=[0,0,0],
        chunk_size=[1, 1, 1],
        volume_size=[1, 1, 28000],
    )
    avg_field_section_vol = CloudVolume(avg_field_section_path, info=avg_field_section_info)
    avg_field_section_vol.commit_info()

    block_dsts = {}
    block_types = ["even", "odd"]
    for i, block_type in enumerate(block_types):
        block_dst = cm.create(
            join(render_dst, "image_blocks", block_type),
            data_type='uint8',
            num_channels=1,
            fill_missing=True,
            overwrite=False,
        )
        block_dsts[i] = block_dst.path

    alignment_z_starts = [args.z_start]
    if args.generate_params_from_skip_file is None:
        last_alignment_start = args.z_start
        while args.z_stop - last_alignment_start > block_size:
            last_alignment_start = last_alignment_start + block_size
            alignment_z_starts.append(last_alignment_start)
    else:
        skip_sections = []
        with open(args.generate_params_from_skip_file) as f:
            line = f.readline()
            while line:
                z = int(line)
                skip_sections.append(z)
                line = f.readline()
        cur_start = args.z_start
        while cur_start < args.z_stop:
            # alignment_z_starts.append(cur_start)
            cur_stop = cur_start + block_size
            prev_stop = cur_stop
            while cur_stop in skip_sections:
                cur_stop = cur_stop + 1
            if cur_stop < args.z_stop:
                alignment_z_starts.append(cur_stop)
            cur_start = prev_stop

    initial_block_starts = [s for s in alignment_z_starts \
                                if (s >= args.z_start and s <= args.z_stop)]
    if initial_block_starts[-1] != args.z_stop:
        initial_block_starts.append(args.z_stop)
    block_starts = []
    for bs, be in zip(initial_block_starts[:-1], initial_block_starts[1:]):
        block_starts.append(bs)
    block_stops = block_starts[1:]
    if block_starts[-1] != args.z_stop:
        block_stops.append(args.z_stop)

    stitch_offset_to_z_range = {i: [] for i in range(1, block_size + 1)}
    block_start_to_stitch_offsets = {i: [] for i in block_starts[1:]}
    for bs, be in zip(block_starts[1:], block_stops[1:]):
        max_offset = 1
        for i, z in enumerate(range(bs, be + 1)):
            if i > 0:
                if len(block_start_to_stitch_offsets[bs]) < max_offset:
                    stitch_offset_to_z_range[i].append(z)
                    block_start_to_stitch_offsets[bs].append(bs - z)
                else:
                    break

    block_dst_lookup = {}
    for k, (bs, be) in enumerate(zip(block_starts, block_stops)):
        even_odd = k % 2
        for i, z in enumerate(range(bs, be + 1)):
            if i > 0:
                block_dst_lookup[z] = block_dsts[even_odd]

    
    stitching_sections = []
    skip_first = True
    got_first = True
    for cur_block_start in block_starts:
        if skip_first and got_first:
            got_first = False
            continue
        block_overlap = args.block_overlap
        if skip_first:
            block_overlap = block_overlap - 1
        stitching_sections.append(cur_block_start + block_overlap)
    
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
            print("Sending {} use time: {}".format(task_iterator, diff))
            if a.distributed:
                print("Run {}".format(task_iterator))
                # wait
                start = time()
                a.wait_for_sqs_empty()
                end = time()
                diff = end - start
                print("Executing {} use time: {}\n".format(task_iterator, diff))

    # Task Scheduling Iterators
    print("Creating task scheduling iterators")


    class AverageFieldChunk:
        def __init__(self, z_range):
            print(z_range)
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                block_dst = block_dst_lookup[z]
                t = a.average_field(
                    cm,
                    bbox,
                    field,
                    mip,
                    avg_field_path,
                    0,
                    bbox,
                    block_dst,
                    # None,
                    pad,
                    src_z=z,
                    dst_z=z,
                )
                yield from t


    class AverageFieldSection:
        def __init__(self, z_range):
            print(z_range)
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                yield tasks.AverageFieldTask(avg_field_path, 0, avg_field_section_path, 
                                            0, avg_chunk_bbox, None, 
                                            pad, z, z)

    class SplitField:
        def __init__(self, z_range):
            print(z_range)
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                temp = avg_field_section_vol[:,:,z].squeeze()
                t = a.split_field(
                    cm,
                    bbox,
                    field,
                    mip,
                    float(temp[0]),
                    float(temp[1]),
                    high_freq_field,
                    mip,
                    src_z=z,
                    dst_z=z,
                )
                yield from t


    # begin_test = 22354
    # end_test = 23354
    # end_test = 22454

    # avg_range = range(args.z_start, args.z_stop, args.block_size)

    # import ipdb
    # ipdb.set_trace()

    # for z_offset in sorted(stitch_offset_to_z_range.keys()):
    #     z_range = list(stitch_offset_to_z_range[z_offset])
    #     for i in range(len(z_range)):
    #         z_range[i] = z_range[i] + args.block_overlap - 1
    #     execute(AverageFieldChunk, z_range)
    # execute(AverageFieldChunk, [22654])
    # execute(AverageFieldSection, [22654])
    if do_average:
        execute(AverageFieldChunk, stitching_sections)
        execute(AverageFieldSection, stitching_sections)
    execute(SplitField, stitching_sections)
    
