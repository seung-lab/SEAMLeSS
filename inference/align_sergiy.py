import gevent.monkey

gevent.monkey.patch_all()

import csv
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from os.path import join
from time import time

from taskqueue import GreenTaskQueue, LocalTaskQueue, MockTaskQueue

from args import get_aligner, get_argparser, get_provenance, parse_args
from boundingbox import BoundingBox
from cloudmanager import CloudManager


def print_run(diff, n_tasks):
    if n_tasks > 0:
        print(
            ": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks)
        )


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
        "--param_lookup",
        type=str,
        help="relative path to CSV file identifying params to use per z range",
    )
    # parser.add_argument('--z_range_path', type=str,
    #   help='path to csv file with list of z indices to use')
    parser.add_argument("--src_path", type=str)
    parser.add_argument(
        "--src_mask_path",
        type=str,
        default="",
        help="CloudVolume path of mask to use with src images; default None",
    )
    parser.add_argument(
        "--src_mask_mip", type=int, default=8, help="MIP of source mask"
    )
    parser.add_argument(
        "--src_mask_val",
        type=int,
        default=1,
        help="Value of of mask that indicates DO NOT mask",
    )
    parser.add_argument("--dst_path", type=str)
    parser.add_argument("--mip", type=int)
    parser.add_argument("--z_start", type=int)
    parser.add_argument("--z_stop", type=int)
    parser.add_argument("--max_mip", type=int, default=9)
    parser.add_argument("--img_dtype", type=str, default='uint8')
    parser.add_argument(
        "--pad",
        help="the size of the largest displacement expected; should be 2^high_mip",
        type=int,
        default=2048,
    )
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--restart", type=int, default=0)
    parser.add_argument(
        "--coarse_field_path",
        type=str,
        help="if specified, applies field to source before aligning to target",
    )
    parser.add_argument(
        "--coarse_field_mip",
        type=int,
        help="MIP level of the primer. E.g. the MIP of a coarse alignment",
    )
    parser.add_argument(
        "--render_dst",
        type=str,
        default=None,
        help="If specified, CloudVolume path to render to instead of default"
    )
    parser.add_argument(
        "--skip_alignment",
        action='store_true',
        help="If True, skip compute field and vector voting"
    )
    parser.add_argument(
        "--skip_render",
        action='store_true',
        help="If True, skip rendering"
    )
    parser.add_argument(
        "--render_mip",
        type=int
    )

    args = parse_args(parser)
    # Only compute matches to previous sections
    args.serial_operation = True
    a = get_aligner(args)
    provenance = get_provenance(args)
    chunk_size = 1024 * 2

    # Simplify var names
    mip = args.mip
    max_mip = args.max_mip
    pad = args.pad
    src_mask_val = args.src_mask_val
    src_mask_mip = args.src_mask_mip
    block_size = args.block_size
    do_alignment = not args.skip_alignment
    do_render = not args.skip_render
    render_mip = args.render_mip or args.mip

    # Create CloudVolume Manager
    cm = CloudManager(
        args.src_path,
        max_mip,
        pad,
        provenance,
        batch_size=1,
        size_chunk=chunk_size,
        batch_mip=mip,
    )
    # Create CloudVolume Manager
    cmr = CloudManager(
        args.src_path,
        max_mip,
        pad,
        provenance,
        batch_size=1,
        size_chunk=chunk_size,
        batch_mip=render_mip,
    )

    # Create src CloudVolumes
    print("Create src & align image CloudVolumes")
    src = cmr.create(
        args.src_path,
        data_type=args.img_dtype,
        num_channels=1,
        fill_missing=True,
        overwrite=False,
    ).path
    src_mask_cv = None
    tgt_mask_cv = None
    if args.src_mask_path:
        src_mask_cv = cm.create(
            args.src_mask_path,
            data_type="uint8",
            num_channels=1,
            fill_missing=True,
            overwrite=False,
        ).path
        tgt_mask_cv = src_mask_cv

    coarse_field_cv = cm.create(
        args.coarse_field_path,
        data_type="int16",
        num_channels=2,
        fill_missing=True,
        overwrite=False,
    ).path
    coarse_field_mip = args.coarse_field_mip
    if coarse_field_mip is None:
        coarse_field_mip = mip




    render_dst = args.dst_path
    if args.render_dst is not None:
        render_dst = args.render_dst

    # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius
    block_dsts = {}
    block_types = ["even", "odd"]
    for i, block_type in enumerate(block_types):
        block_dst = cmr.create(
            join(render_dst, "image_blocks", block_type),
            data_type=args.img_dtype,
            num_channels=1,
            fill_missing=True,
            overwrite=do_render,
        )
        block_dsts[i] = block_dst.path

    # import ipdb
    # ipdb.set_trace

    # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
    bbox_lookup = {}
    model_lookup = {}
    tgt_radius_lookup = {}
    vvote_lookup = {}
    # skip_list = []
    skip_list = [17491, 17891]
    with open(args.param_lookup) as f:
        reader = csv.reader(f, delimiter=",")
        for k, r in enumerate(reader):
            if k != 0:
                x_start = int(r[0])
                y_start = int(r[1])
                z_start = int(r[2])
                x_stop = int(r[3])
                y_stop = int(r[4])
                z_stop = int(r[5])
                bbox_mip = int(r[6])
                model_path = join("..", "models", r[7])
                tgt_radius = int(r[8])
                skip = bool(int(r[9]))
                bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
                # print('{},{}'.format(z_start, z_stop))
                for z in range(z_start, z_stop):
                    if skip:
                        skip_list.append(z)
                    bbox_lookup[z] = bbox
                    model_lookup[z] = model_path
                    tgt_radius_lookup[z] = tgt_radius
                    vvote_lookup[z] = [-i for i in range(1, tgt_radius + 1)]

    # Filter out skipped sections from vvote_offsets
    min_offset = 0
    for z, tgt_radius in vvote_lookup.items():
        offset = 0
        for i, r in enumerate(tgt_radius):
            while r + offset + z in skip_list:
                offset -= 1
            tgt_radius[i] = r + offset
        min_offset = min(min_offset, r + offset)
        offset = 0
        vvote_lookup[z] = tgt_radius

    # Adjust block starts so they don't start on a skipped section
    initial_block_starts = list(range(args.z_start, args.z_stop, block_size))
    if initial_block_starts[-1] != args.z_stop:
        initial_block_starts.append(args.z_stop)
    block_starts = []
    for bs, be in zip(initial_block_starts[:-1], initial_block_starts[1:]):
        while bs in skip_list:
            bs += 1
            assert bs < be
        block_starts.append(bs)
    block_stops = block_starts[1:]
    if block_starts[-1] != args.z_stop:
        block_stops.append(args.z_stop)
    section_to_block_start = {}
    for z in range(args.z_start, args.z_stop):
        # try:
        section_to_block_start[z] = max(filter(lambda x: (x <= z), block_starts))
        # except:
            # import ipdb
            # ipdb.set_trace()
    # print('initial_block_starts {}'.format(list(initial_block_starts)))
    # print('block_starts {}'.format(block_starts))
    # print('block_stops {}'.format(block_stops))
    # Assign even/odd to each block start so results are stored in appropriate CloudVolume
    # Create lookup dicts based on offset in the canonical block
    # BLOCK ALIGNMENT
    # Copy sections with block offsets of 0
    # Align without vector voting sections with block offsets < 0 (starter sections)
    # Align with vector voting sections with block offsets > 0 (block sections)
    # This lookup makes it easy for restarting based on block offset, though isn't
    #  strictly necessary for the copy & starter sections
    # BLOCK STITCHING
    # Stitch blocks using the aligned block sections that have tgt_z in the starter sections
    block_dst_lookup = {}
    block_start_lookup = {}
    starter_dst_lookup = {}
    copy_offset_to_z_range = {0: deepcopy(block_starts)}
    overlap_copy_range = set()
    starter_offset_to_z_range = {i: set() for i in range(min_offset, 0)}
    block_offset_to_z_range = {
        i: set() for i in range(1, block_size + 10)
    }  # TODO: Set the padding based on max(be-bs)
    # Reverse lookup to easily identify tgt_z for each starter z
    starter_z_to_offset = {}
    for k, (bs, be) in enumerate(zip(block_starts, block_stops)):
        even_odd = k % 2
        for i, z in enumerate(range(bs, be + 1)):
            if i > 0:
                block_start_lookup[z] = bs
                block_dst_lookup[z] = block_dsts[even_odd]
                if z not in skip_list:
                    block_offset_to_z_range[i].add(z)
                    for tgt_offset in vvote_lookup[z]:
                        tgt_z = z + tgt_offset
                        if tgt_z <= bs:
                            starter_dst_lookup[tgt_z] = block_dsts[even_odd]
                            # ignore first block for stitching operations
                            if k > 0:
                                overlap_copy_range.add(tgt_z)
                        if tgt_z < bs:
                            starter_z_to_offset[tgt_z] = bs - tgt_z
                            starter_offset_to_z_range[tgt_z - bs].add(tgt_z)
    offset_range = [i for i in range(min_offset, abs(min_offset) + 1)]
    # check for restart
    print("Align starting from OFFSET {}".format(args.restart))
    starter_restart = -100
    if args.restart <= 0:
        starter_restart = args.restart

    original_copy_range = [z for z_range in copy_offset_to_z_range.values() for z in z_range]
    original_starter_range = [
        z for z_range in starter_offset_to_z_range.values() for z in z_range
    ]

    copy_offset_to_z_range = {
        k: v for k, v in copy_offset_to_z_range.items() if k == args.restart
    }
    starter_offset_to_z_range = {
        k: v for k, v in starter_offset_to_z_range.items() if k <= starter_restart
    }
    block_offset_to_z_range = {
        k: v for k, v in block_offset_to_z_range.items() if k >= args.restart
    }
    # print('copy_offset_to_z_range {}'.format(copy_offset_to_z_range))
    # print('starter_offset_to_z_range {}'.format(starter_offset_to_z_range))
    # print('block_offset_to_z_range {}'.format(block_offset_to_z_range))
    # print('offset_range {}'.format(offset_range))
    copy_range = [z for z_range in copy_offset_to_z_range.values() for z in z_range]
    starter_range = [
        z for z_range in starter_offset_to_z_range.values() for z in z_range
    ]
    overlap_copy_range = list(overlap_copy_range)
    # print('overlap_copy_range {}'.format(overlap_copy_range))

    # Determine the number of sections needed to stitch (no stitching for block 0)
    stitch_offset_to_z_range = {i: [] for i in range(1, block_size + 1)}
    block_start_to_stitch_offsets = {i: [] for i in block_starts[1:]}
    for bs, be in zip(block_starts[1:], block_stops[1:]):
        max_offset = 0
        for i, z in enumerate(range(bs, be + 1)):
            if i > 0 and z not in skip_list:
                max_offset = max(max_offset, tgt_radius_lookup[z])
                if len(block_start_to_stitch_offsets[bs]) < max_offset:
                    stitch_offset_to_z_range[i].append(z)
                    block_start_to_stitch_offsets[bs].append(bs - z)
                else:
                    break
    stitch_range = [z for z_range in stitch_offset_to_z_range.values() for z in z_range]
    for b, v in block_start_to_stitch_offsets.items():
        print(b)
        assert len(v) % 2 == 1

    # Create field CloudVolumes
    print("Creating field & overlap CloudVolumes")
    block_pair_fields = {}
    for z_offset in offset_range:
        block_pair_fields[z_offset] = cm.create(
            join(args.dst_path, "field", "block", str(z_offset)),
            data_type="int16",
            num_channels=2,
            fill_missing=True,
            overwrite=do_alignment,
        ).path
    block_vvote_field = cm.create(
        join(args.dst_path, "field", "vvote"),
        data_type="int16",
        num_channels=2,
        fill_missing=True,
        overwrite=do_alignment,
    ).path
    stitch_pair_fields = {}
    for z_offset in offset_range:
        stitch_pair_fields[z_offset] = cm.create(
            join(args.dst_path, "field", "stitchtemp", str(z_offset)),
            data_type="int16",
            num_channels=2,
            fill_missing=True,
            overwrite=do_alignment,
        ).path
    overlap_vvote_field = cm.create(
        join(args.dst_path, "field", "stitchtemp", "vvote", "field"),
        data_type="int16",
        num_channels=2,
        fill_missing=True,
        overwrite=do_alignment,
    ).path
    overlap_image = cm.create(
        join(args.dst_path, "field", "stitchtemp", "vvote", "image"),
        data_type=args.img_dtype,
        num_channels=1,
        fill_missing=True,
        overwrite=do_render,
    ).path
    stitch_fields = {}
    for z_offset in offset_range:
        stitch_fields[z_offset] = cm.create(
            join(args.dst_path, "field", "stitchtemp", "vvote", str(z_offset)),
            data_type="int16",
            num_channels=2,
            fill_missing=True,
            overwrite=do_alignment,
        ).path
    broadcasting_field = cm.create(
        join(args.dst_path, "field", "stitchtemp", "broadcasting"),
        data_type="int16",
        num_channels=2,
        fill_missing=True,
        overwrite=do_alignment,
    ).path

    # import ipdb
    # ipdb.set_trace()

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
                if do_alignment:
                    a.wait_for_sqs_empty()
                end = time()
                diff = end - start
                print("Executing {} use time: {}\n".format(task_iterator, diff))

    # Task Scheduling Iterators
    print("Creating task scheduling iterators")

    class StarterCopy:
        def __init__(self, z_range):
            print(z_range)
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                block_dst = starter_dst_lookup[z]
                bbox = bbox_lookup[z]
                # t =  a.copy(cm, src, block_dst, z, z, bbox, mip, is_field=False,
                #             mask_cv=src_mask_cv, mask_mip=src_mask_mip, mask_val=src_mask_val)

                t = a.render(
                    cm,
                    src,
                    coarse_field_cv,
                    block_dst,
                    src_z=z,
                    field_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=render_mip,
                    field_mip=coarse_field_mip,
                    mask_cv=src_mask_cv,
                    mask_val=src_mask_val,
                    mask_mip=src_mask_mip,
                )
                yield from t

    class StarterUpsampleField:
        def __init__(self, z_range):
            print(z_range)
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                bbox = bbox_lookup[z]
                field_dst = block_pair_fields[0]

                t = a.cloud_upsample_field(
                    cm,
                    coarse_field_cv,
                    field_dst,
                    src_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=coarse_field_mip,
                    dst_mip=mip
                )
                yield from t

    class StarterComputeField(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                dst = starter_dst_lookup[z]
                model_path = model_lookup[z]
                bbox = bbox_lookup[z]
                z_offset = starter_z_to_offset[z]
                field = block_pair_fields[z_offset]
                tgt_field = block_pair_fields[0]
                tgt_z = z + z_offset
                t = a.compute_field(
                    cm,
                    model_path,
                    src,
                    dst,
                    field,
                    z,
                    tgt_z,
                    bbox,
                    mip,
                    pad,
                    src_mask_cv=src_mask_cv,
                    src_mask_mip=src_mask_mip,
                    src_mask_val=src_mask_val,
                    tgt_mask_cv=src_mask_cv,
                    tgt_mask_mip=src_mask_mip,
                    tgt_mask_val=src_mask_val,
                    prev_field_cv=None,
                    prev_field_z=None,
                    coarse_field_cv=coarse_field_cv,
                    coarse_field_mip=coarse_field_mip,
                    tgt_field_cv=tgt_field,
                )
                yield from t

    class StarterRender(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                dst = starter_dst_lookup[z]
                z_offset = starter_z_to_offset[z]
                fine_field = block_pair_fields[z_offset]
                bbox = bbox_lookup[z]
                t = a.render(
                    cm,
                    src,
                    fine_field,
                    dst,
                    src_z=z,
                    field_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=render_mip,
                    field_mip=mip,
                    mask_cv=src_mask_cv,
                    mask_val=src_mask_val,
                    mask_mip=src_mask_mip,
                )
                yield from t

    class BlockAlignComputeField(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for src_z in self.z_range:
                dst = block_dst_lookup[src_z]
                bbox = bbox_lookup[src_z]
                model_path = model_lookup[src_z]
                tgt_offsets = vvote_lookup[src_z]
                for tgt_offset in tgt_offsets:
                    tgt_z = src_z + tgt_offset
                    fine_field = block_pair_fields[tgt_offset]
                    if tgt_z in original_copy_range:
                        tgt_field = block_pair_fields[0]
                    elif tgt_z in original_starter_range and src_z > section_to_block_start[src_z] and section_to_block_start[src_z] > tgt_z:
                        tgt_field = block_pair_fields[starter_z_to_offset[tgt_z]]
                    else:
                        tgt_field = block_vvote_field
                    t = a.compute_field(
                        cm,
                        model_path,
                        src,
                        dst,
                        fine_field,
                        src_z,
                        tgt_z,
                        bbox,
                        mip,
                        pad,
                        src_mask_cv=src_mask_cv,
                        src_mask_mip=src_mask_mip,
                        src_mask_val=src_mask_val,
                        tgt_mask_cv=src_mask_cv,
                        tgt_mask_mip=src_mask_mip,
                        tgt_mask_val=src_mask_val,
                        prev_field_cv=tgt_field,
                        prev_field_z=tgt_z,
                        coarse_field_cv=coarse_field_cv,
                        coarse_field_mip=coarse_field_mip,
                        tgt_field_cv=tgt_field,
                    )
                    yield from t

    class BlockAlignVectorVote(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                bbox = bbox_lookup[z]
                tgt_offsets = vvote_lookup[z]
                fine_fields = {i: block_pair_fields[i] for i in tgt_offsets}
                t = a.vector_vote(
                    cm,
                    fine_fields,
                    block_vvote_field,
                    z,
                    bbox,
                    mip,
                    inverse=False,
                    serial=True,
                    softmin_temp=(2 ** coarse_field_mip) / 6.0,
                    blur_sigma=1,
                )
                yield from t

    class BlockAlignRender(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                dst = block_dst_lookup[z]
                bbox = bbox_lookup[z]
                t = a.render(
                    cm,
                    src,
                    block_vvote_field,
                    dst,
                    src_z=z,
                    field_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=render_mip,
                    field_mip=mip,
                    mask_cv=src_mask_cv,
                    mask_val=src_mask_val,
                    mask_mip=src_mask_mip,
                )
                yield from t

    class StitchOverlapCopy:
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                dst = block_dst_lookup[z]
                bbox = bbox_lookup[z]
                ti = a.copy(cm, dst, overlap_image, z, z, bbox, mip, is_field=False)
                tf = a.copy(
                    cm,
                    block_vvote_field,
                    overlap_vvote_field,
                    z,
                    z,
                    bbox,
                    mip,
                    is_field=True,
                )
                t = ti + tf
                yield from t

    # class StitchAlignComputeField(object):
    #     def __init__(self, z_range):
    #         self.z_range = z_range

    #     def __iter__(self):
    #         for z in self.z_range:
    #             block_dst = block_dst_lookup[z]
    #             bbox = bbox_lookup[z]
    #             model_path = model_lookup[z]
    #             tgt_offsets = vvote_lookup[z]
    #             last_tgt_offset = tgt_offsets[0] + 1 # HACK
    #             for tgt_offset in tgt_offsets:
    #                 tgt_z = z + tgt_offset
    #                 fine_field = stitch_pair_fields[tgt_offset]
    #                 if last_tgt_offset > 0:
    #                     tgt_field = overlap_vvote_field
    #                 else:
    #                     tgt_field = stitch_pair_fields[last_tgt_offset]
    #                 last_tgt_offset = tgt_offset
    #                 t = a.compute_field(
    #                     cm,
    #                     model_path,
    #                     src,
    #                     overlap_image,
    #                     fine_field,
    #                     z,
    #                     tgt_z,
    #                     bbox,
    #                     mip,
    #                     pad,
    #                     src_mask_cv=src_mask_cv,
    #                     src_mask_mip=src_mask_mip,
    #                     src_mask_val=src_mask_val,
    #                     tgt_mask_cv=src_mask_cv,
    #                     tgt_mask_mip=src_mask_mip,
    #                     tgt_mask_val=src_mask_val,
    #                     tgt_field_cv=block_vvote_field,
    #                     stitch=True
    #                 )
    #                 yield from t

    class StitchAlignComputeField(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                block_dst = block_dst_lookup[z]
                bbox = bbox_lookup[z]
                model_path = model_lookup[z]
                tgt_offsets = vvote_lookup[z]
                for tgt_offset in tgt_offsets:
                    tgt_z = z + tgt_offset
                    field = stitch_pair_fields[tgt_offset]
                    t = a.compute_field(cm, model_path, block_dst, overlap_image, field,
                                        z, tgt_z, bbox, mip, pad, src_mask_cv=src_mask_cv,
                                        src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                                        tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip,
                                        tgt_mask_val=src_mask_val,
                                        prev_field_cv=overlap_vvote_field,
                                        # prev_field_cv=None,
                                        prev_field_z=tgt_z,stitch=True)
                    yield from t

    # class StitchAlignComputeField(object):
    #     def __init__(self, z_range):
    #         self.z_range = z_range

    #     def __iter__(self):
    #         for z in self.z_range:
    #             block_dst = block_dst_lookup[z]
    #             bbox = bbox_lookup[z]
    #             model_path = model_lookup[z]
    #             tgt_offsets = vvote_lookup[z]
    #             last_tgt_offset = tgt_offsets[0] + 1 # HACK
    #             for tgt_offset in tgt_offsets:
    #                 tgt_z = z + tgt_offset
    #                 fine_field = stitch_pair_fields[tgt_offset]
    #                 if last_tgt_offset > 0:
    #                     tgt_field = overlap_vvote_field
    #                 else:
    #                     tgt_field = stitch_pair_fields[last_tgt_offset]
    #                 last_tgt_offset = tgt_offset
    #                 t = a.compute_field(
    #                     cm,
    #                     model_path,
    #                     block_dst,
    #                     overlap_image,
    #                     fine_field,
    #                     z,
    #                     tgt_z,
    #                     bbox,
    #                     mip,
    #                     pad,
    #                     src_mask_cv=src_mask_cv,
    #                     src_mask_mip=src_mask_mip,
    #                     src_mask_val=src_mask_val,
    #                     tgt_mask_cv=src_mask_cv,
    #                     tgt_mask_mip=src_mask_mip,
    #                     tgt_mask_val=src_mask_val,
    #                     prev_field_cv=overlap_vvote_field,
    #                     prev_field_z=tgt_z,
    #                     coarse_field_cv=coarse_field_cv,
    #                     coarse_field_mip=coarse_field_mip,
    #                     tgt_field_cv=tgt_field,
    #                 )
    #                 yield from t

    # class StitchAlignComputeField(object):
    #     def __init__(self, z_range):
    #         self.z_range = z_range

    #     def __iter__(self):
    #         for z in self.z_range:
    #             block_dst = block_dst_lookup[z]
    #             bbox = bbox_lookup[z]
    #             model_path = model_lookup[z]
    #             tgt_offsets = vvote_lookup[z]
    #             last_tgt_offset = tgt_offsets[0] + 1 # HACK
    #             for tgt_offset in tgt_offsets:
    #                 tgt_z = z + tgt_offset
    #                 fine_field = stitch_pair_fields[tgt_offset]
    #                 if last_tgt_offset > 0:
    #                     tgt_field = overlap_vvote_field
    #                 else:
    #                     tgt_field = stitch_pair_fields[last_tgt_offset]
    #                 last_tgt_offset = tgt_offset
    #                 t = a.compute_field(
    #                     cm,
    #                     model_path,
    #                     src,
    #                     overlap_image,
    #                     fine_field,
    #                     z,
    #                     tgt_z,
    #                     bbox,
    #                     mip,
    #                     pad,
    #                     src_mask_cv=src_mask_cv,
    #                     src_mask_mip=src_mask_mip,
    #                     src_mask_val=src_mask_val,
    #                     tgt_mask_cv=src_mask_cv,
    #                     tgt_mask_mip=src_mask_mip,
    #                     tgt_mask_val=src_mask_val,
    #                     tgt_field_cv=block_vvote_field,
    #                     stitch=True
    #                 )
    #                 yield from t

    class StitchAlignVectorVote(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                bbox = bbox_lookup[z]
                tgt_offsets = vvote_lookup[z]
                fine_fields = {i: stitch_pair_fields[i] for i in tgt_offsets}
                t = a.vector_vote(
                    cm,
                    fine_fields,
                    overlap_vvote_field,
                    z,
                    bbox,
                    mip,
                    inverse=False,
                    serial=True,
                    # softmin_temp=(2 ** coarse_field_mip) / 6.0,
                    softmin_temp=(2 ** mip) / 6.0,
                    blur_sigma=1,
                )
                yield from t

    class StitchAlignRender(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                block_dst = block_dst_lookup[z]
                bbox = bbox_lookup[z]
                t = a.render(
                    cm,
                    block_dst,
                    overlap_vvote_field,
                    overlap_image,
                    src_z=z,
                    field_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=render_mip,
                    field_mip=mip,
                    mask_cv=src_mask_cv,
                    mask_val=src_mask_val,
                    mask_mip=src_mask_mip,
                )
                yield from t

    class StitchBroadcastCopy:
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                bs = block_start_lookup[z]
                z_offset = bs - z
                stitch_field = stitch_fields[z_offset]
                bbox = bbox_lookup[z]
                t = a.copy(
                    cm,
                    overlap_vvote_field,
                    stitch_field,
                    z,
                    bs,
                    bbox,
                    mip,
                    is_field=True,
                )
                yield from t

    class StitchBroadcastVectorVote(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                bbox = bbox_lookup[z]
                offsets = block_start_to_stitch_offsets[z]
                fields = {i: stitch_fields[i] for i in offsets}
                t = a.vector_vote(
                    cm,
                    fields,
                    broadcasting_field,
                    z,
                    bbox,
                    mip,
                    inverse=False,
                    serial=True,
                    # softmin_temp=(2 ** coarse_field_mip) / 6.0,
                    softmin_temp=(2 ** mip) / 6.0,
                    blur_sigma=1,
                )
                yield from t

    # # Serial alignment with block stitching
    print("START BLOCK ALIGNMENT")
    if do_render:
        print("COPY STARTING SECTION OF ALL BLOCKS")
        execute(StarterCopy, copy_range)
    if do_alignment:
        if coarse_field_cv is not None:
            print("UPSAMPLE STARTING SECTION COARSE FIELDS OF ALL BLOCKS")
            execute(StarterUpsampleField, copy_range)
        print("ALIGN STARTER SECTIONS FOR EACH BLOCK")
        execute(StarterComputeField, starter_range)
    if do_render:
        execute(StarterRender, starter_range)
    for z_offset in sorted(block_offset_to_z_range.keys()):
        z_range = list(block_offset_to_z_range[z_offset])
        if do_alignment:
            print("ALIGN BLOCK OFFSET {}".format(z_offset))
            execute(BlockAlignComputeField, z_range)
            print("VECTOR VOTE BLOCK OFFSET {}".format(z_offset))
            execute(BlockAlignVectorVote, z_range)
        if do_render:
            print("RENDER BLOCK OFFSET {}".format(z_offset))
            execute(BlockAlignRender, z_range)

    print("END BLOCK ALIGNMENT")
    print("START BLOCK STITCHING")
    print("COPY OVERLAPPING IMAGES & FIELDS OF BLOCKS")
    if do_render:
        execute(StitchOverlapCopy, overlap_copy_range)
    for z_offset in sorted(stitch_offset_to_z_range.keys()):
        z_range = list(stitch_offset_to_z_range[z_offset])
        if do_alignment:
            print("ALIGN OVERLAPPING OFFSET {}".format(z_offset))
            execute(StitchAlignComputeField, z_range)
            print("VECTOR VOTE OVERLAPPING OFFSET {}".format(z_offset))
            execute(StitchAlignVectorVote, z_range)
        if do_render:
            print("RENDER OVERLAPPING OFFSET {}".format(z_offset))
            execute(StitchAlignRender, z_range)

    if do_alignment:
        print("COPY OVERLAP ALIGNED FIELDS FOR VECTOR VOTING")
        execute(StitchBroadcastCopy, stitch_range)
        print("VECTOR VOTE STITCHING FIELDS")
        execute(StitchBroadcastVectorVote, block_starts[1:])

