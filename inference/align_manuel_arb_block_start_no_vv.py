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


if __name__ == "__main__":
    parser = get_argparser()
    parser.add_argument(
        "--param_lookup",
        type=str,
        help="relative path to CSV file identifying params to use per z range",
    )
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument('--src_mask', action='append',
            help='Pass string that contains a JSON dict. Fields: "cv", "mip", "val", "op"',
            type=json.loads, dest='src_masks')
    parser.add_argument('--tgt_mask', action='append',
            help='Pass string that contains a JSON dict. Fields: "cv", "mip", "val", "op"',
            type=json.loads, dest='tgt_masks')
    parser.add_argument("--dst_path", type=str)
    parser.add_argument("--mip", type=int)
    parser.add_argument("--z_start", type=int)
    parser.add_argument("--z_stop", type=int)
    parser.add_argument("--max_mip", type=int, default=9)
    parser.add_argument("--img_dtype", type=str, default='uint8')
    parser.add_argument("--final_render_pad", type=int, default=2048)
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
        "--coarse_field_dtype",
        type=str,
        help="Data type of coarse vector field (typically int16 or float32)",
        default='int16'
    )
    parser.add_argument(
        "--output_field_dtype",
        type=str,
        help="Data type of output vector fields (typically int16 or float32)",
        default='int16'
    )
    parser.add_argument(
        "--skip_alignment",
        action='store_true',
        help="If True, skip compute field and vector voting"
    )
    parser.add_argument(
        "--skip_final_render",
        action='store_true',
        help="If True, skip final render"
    )
    parser.add_argument(
        "--skip_stitching",
        action='store_true',
        help="If True, skip stitching"
    )
    parser.add_argument(
        "--skip_compose",
        action='store_true',
        help="If True, skip composition"
    )
    parser.add_argument(
        "--skip_render",
        action='store_true',
        help="If True, skip rendering"
    )
    parser.add_argument(
        "--final_render_mip",
        type=int,
        default=None
    )
    parser.add_argument(
        "--decay_dist",
        type=int,
        default=None
    )
    parser.add_argument(
        "--seethrough",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--seethrough_misalign",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--blackout_op",
        type=str,
        default='none'
    )
    parser.add_argument('--stitch_suffix', type=str, default='', help='string to append to directory names')
    parser.add_argument('--status_output_file', type=str, default=None)
    parser.add_argument('--recover_status_from_file', type=str, default=None)
    parser.add_argument(
        "--block_overlap",
        type=int,
        default=0
    )
    parser.add_argument('--skip_list_lookup', type=str, help='relative path to file identifying list of skip sections')

    args = parse_args(parser)
    # Only compute matches to previous sections
    args.serial_operation = True
    a = get_aligner(args)
    provenance = get_provenance(args)
    chunk_size = args.chunk_size
    # Simplify var names
    mip = args.mip
    max_mip = args.max_mip
    pad = args.pad
    final_render_pad = args.final_render_pad or args.pad
    src_masks = []
    tgt_masks = []
    if args.src_masks is not None:
        src_masks = [Mask(**m) for m in args.src_masks]
    if args.tgt_masks is not None:
        tgt_masks = [Mask(**m) for m in args.tgt_masks]

    block_size = args.block_size
    do_alignment = not args.skip_alignment
    do_render = not args.skip_render
    do_stitching = not args.skip_stitching
    do_compose = not args.skip_compose
    do_final_render = not args.skip_final_render
    blackout_op = args.blackout_op
    stitch_suffix = args.stitch_suffix
    output_field_dtype = args.output_field_dtype

    final_render_mip = args.final_render_mip or args.mip
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

    # Create src CloudVolumes
    print("Create src & align image CloudVolumes")
    src = cm.create(
        args.src_path,
        data_type=args.img_dtype,
        num_channels=1,
        fill_missing=True,
        overwrite=False,
    ).path

    for mask in (src_masks + tgt_masks):
        cm.create(
            mask.cv_path,
            data_type=mask.dtype,
            num_channels=1,
            fill_missing=True,
            overwrite=False,
        )


    coarse_field_cv = cm.create(
        args.coarse_field_path,
        data_type=args.coarse_field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=False,
    ).path
    coarse_field_mip = args.coarse_field_mip
    if coarse_field_mip is None:
        coarse_field_mip = mip


    # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius
    block_dsts = {}
    # block_field_dsts = {}
    block_types = ["even", "odd"]
    for i, block_type in enumerate(block_types):
        block_dst = cm.create(
            join(args.dst_path, "image_blocks", block_type),
            data_type=args.img_dtype,
            num_channels=1,
            fill_missing=True,
            overwrite=do_render,
        )
        block_dsts[i] = block_dst.path
        # # block_field = cm.create(
        #     # join(args.dst_path, "field", block_type),
        #     data_type=output_field_dtype,
        #     num_channels=2,
        #     fill_missing=True,
        #     overwrite=do_alignment,
        # )
        # block_field_dsts[i] = block_field.path

    # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
    bbox_lookup = {}
    model_lookup = {}
    block_alignment_zs = []
    alignment_z_starts = []
    stitching_sections = []
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
                alignment_z_starts.append(z_start)
                bbox_mip = int(r[6])
                model_path = join("..", "models", r[7])
                bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
                for z in range(z_start, z_stop):
                    bbox_lookup[z] = bbox
                    model_lookup[z] = model_path

    for i in range(len(alignment_z_starts) - 1):
        cur_start = alignment_z_starts[i]
        next_start = alignment_z_starts[i+1]
        block_alignment_zs.append([*range(cur_start+1, next_start + 1 + args.block_overlap)])
        stitching_sections.append(next_start + 1 + args.block_overlap)
    
    block_alignment_zs.append([*range(z_starts[-1]+1, args.z_stop)])

    block_dst_lookup = {}
    block_start_lookup = {}
    starter_dst_lookup = {}

    even_or_odd = 0
    for i in range(len(alignment_z_starts)):
        starter_dst_lookup[alignment_z_starts] = block_dsts[even_or_odd]
        for alignment_z in alignment_z_starts[i]:
            block_start_lookup[alignment_z] = alignment_z_starts[i]
            block_dst_lookup[alignment_z] = block_dsts[even_or_odd]
        even_or_odd = (even_or_odd + 1) % 2

    # Create field CloudVolumes
    print("Creating field & overlap CloudVolumes")
    stitch_field = cm.create(
        join(args.dst_path, "field", "stitch"),
        data_type=output_field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=do_alignment,
    ).path
    compose_field = cm.create(join(args.dst_path, 'field',
                        'stitch{}'.format(args.stitch_suffix), 'compose'),
                        data_type=output_field_dtype, num_channels=2,
                        fill_missing=True, overwrite=do_compose).path
    if do_final_render:
        # Create CloudVolume Manager
        cmr = CloudManager(
            args.src_path,
            max_mip,
            pad,
            provenance,
            batch_size=1,
            size_chunk=chunk_size,
            batch_mip=final_render_mip,
        )
        final_dst = cmr.create(join(args.dst_path, 'image_stitch{}'.format(args.stitch_suffix)),
                            data_type='uint8', num_channels=1, fill_missing=True,
                            overwrite=do_final_render).path

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
                t = a.render(
                    cm,
                    src,
                    coarse_field_cv,
                    block_dst,
                    src_z=z,
                    field_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=mip,
                    field_mip=coarse_field_mip,
                    masks=src_masks,
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
                    src_masks=src_masks,
                    tgt_masks=tgt_masks,
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
                    # src_mip=render_mip,
                    src_mip=mip,
                    field_mip=mip,
                    masks=src_masks,
                    seethrough=args.seethrough,
                    seethrough_misalign=args.seethrough_misalign
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
                    fine_field = block_vvote_field
                    if tgt_z in copy_range:
                        tgt_field = block_pair_fields[0]
                    elif tgt_z in starter_range and src_z > block_start_lookup[src_z] and block_start_lookup[src_z] > tgt_z:
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
                        src_masks=src_masks,
                        tgt_masks=tgt_masks,
                        prev_field_cv=tgt_field,
                        prev_field_z=tgt_z,
                        coarse_field_cv=coarse_field_cv,
                        coarse_field_mip=coarse_field_mip,
                        tgt_field_cv=tgt_field,
                        report=True
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
                    src_mip=mip,
                    field_mip=mip,
                    masks=src_masks,
                    pad=pad,
                    seethrough=args.seethrough,
                    seethrough_misalign=args.seethrough_misalign,
                    report=True
                )
                yield from t

    # seethrough_offset = 5

    # class SeethroughStitchRender(object):
    #     def __init__(self, z_range):
    #         self.z_range = z_range

    #     def __iter__(self):
    #         for z in self.z_range:
    #             z_start = z
    #             z_end = z + seethrough_offset

    #             src = block_dst_lookup[z]
    #             dst = seethrough_stitch_dst
    #             bbox = bbox_lookup[z]
    #             t = a.seethrough_stitch_render(
    #                 cm,
    #                 src,
    #                 dst,
    #                 z_start=z_start,
    #                 z_end=z_end,
    #                 bbox=bbox,
    #                 mip=mip,
    #                 pad=pad
    #             )

    #             yield from t


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
                    fields = broadcasting_field
                    starter_z = block_start_lookup[z]
                    t = a.compute_field(cm, model_path, block_dst, overlap_image, field,
                                        starter_z, tgt_z, bbox, mip, pad,
                                        src_masks=src_masks,
                                        tgt_masks=tgt_masks,
                                        prev_field_cv=None,
                                        prev_field_z=tgt_z,stitch=True)
                    yield from t

    class StitchCompose(object):
        def __init__(self, z_range):
          self.z_range = z_range

        def __iter__(self):
          for z in self.z_range:
            influencing_blocks = influencing_blocks_lookup[z]
            factors = [interpolate(z, bs, decay_dist) for bs in influencing_blocks]
            factors += [1.]
            print('z={}\ninfluencing_blocks {}\nfactors {}'.format(z, influencing_blocks,
                                                                   factors))
            bbox = bbox_lookup[z]
            cv_list = [broadcasting_field]*len(influencing_blocks) + [block_vvote_field]
            z_list = list(influencing_blocks) + [z]
            t = a.multi_compose(cm, cv_list, compose_field, z_list, z, bbox,
                                mip, mip, factors, pad)
            yield from t

    class StitchFinalRender(object):    
        def __init__(self, z_range):
          self.z_range = z_range

        def __iter__(self):
          for z in self.z_range:
            bbox = bbox_lookup[z]
            t = a.render(cm, src, compose_field, final_dst, src_z=z,
                         field_z=z, dst_z=z, bbox=bbox,
                         src_mip=final_render_mip, field_mip=mip, pad=final_render_pad,
                         masks=src_masks, blackout_op=blackout_op)
            yield from t

    # compute_field_map = {}
    # render_map = {}
    max_dist = 1
    block_z_list = []
    first_z_release = None
    for z_offset in sorted(block_offset_to_z_range.keys()):
        if first_z_release is None:
            first_z_release = list(block_offset_to_z_range[z_offset])
        block_z_list.append(list(block_offset_to_z_range[z_offset]))
    block_z_list = np.concatenate(block_z_list)

    def break_into_chunks(chunk_size, offset, mip, max_mip=12):
        """Break bbox into list of chunks with chunk_size, given offset for all data

        Args:
        bbox: BoundingBox for region to be broken into chunks
        chunk_size: tuple for dimensions of chunk that bbox will be broken into;
            will be set to min(chunk_size, self.chunk_size)
        offset: tuple for x,y origin for the entire dataset, from which chunks
            will be aligned
        mip: int for MIP level at which bbox is defined
        max_mip: int for the maximum MIP level at which the bbox is valid
        """
        chunks = []
        z_to_number_of_chunks = {}
        for z in block_z_list:
            bbox = bbox_lookup[z]
            raw_x_range = bbox.x_range(mip=mip)
            raw_y_range = bbox.y_range(mip=mip)

            x_chunk = chunk_size[0]
            y_chunk = chunk_size[1]

            x_offset = offset[0]
            y_offset = offset[1]
            x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
            y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

            calign_x_range = [raw_x_range[0] - x_remainder, raw_x_range[1]]
            calign_y_range = [raw_y_range[0] - y_remainder, raw_y_range[1]]

            x_size = len(range(calign_x_range[0], calign_x_range[1], chunk_size[0]))
            y_size = len(range(calign_y_range[0], calign_y_range[1], chunk_size[0]))

            z_to_number_of_chunks[z] = x_size * y_size

            for xs in range(calign_x_range[0], calign_x_range[1], chunk_size[0]):
                for ys in range(calign_y_range[0], calign_y_range[1], chunk_size[1]):
                    chunks.append((xs, ys, int(z)))
                    # chunks.append(BoundingBox(xs, xs + chunk_size[0],
                                            #  ys, ys + chunk_size[1],
                                            #  mip=mip, max_mip=max_mip))
        return chunks, z_to_number_of_chunks

    chunks, z_to_number_of_chunks = break_into_chunks(cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, max_mip=cm.max_mip)
    chunk_to_compute_processed = dict(zip(chunks, [False] * len(chunks)))
    chunk_to_render_processed = dict(zip(chunks, [False] * len(chunks)))
    z_to_computes_processed = dict(zip(block_z_list, [0] * len(chunks)))
    z_to_renders_processed = dict(zip(block_z_list, [0] * len(chunks)))
    # z_to_chunks_processed = dict(zip(block_z_list, [0] * len(block_z_list)))
    z_to_compute_released = dict(zip(block_z_list, [False] * len(block_z_list)))
    z_to_render_released = dict(zip(block_z_list, [False] * len(block_z_list)))
    renders_complete = 0

    def recover_status_from_file(filename):
        global renders_complete
        with open(filename, 'r') as recover_file:
            line = recover_file.readline()
            while line:
                z = int(line[3:])
                if line[0:2] == 'cf':
                    z_to_compute_released[z] = True
                elif line[0:2] == 'rt':
                    z_to_render_released[z] = True
                    renders_complete = renders_complete + 1
                line = recover_file.readline()
        new_cf_list = []
        new_rt_list = []
        for z in first_z_release:
            # import ipdb
            # ipdb.set_trace()
            while z in z_to_compute_released:
                if not z_to_compute_released[z]:
                    new_cf_list.append(z)
                    break
                if not z_to_render_released[z]:
                    new_rt_list.append(z)
                    break
                z = z + 1
        return new_cf_list, new_rt_list

    import ipdb
    ipdb.set_trace()

    def executeNew(task_iterator, z_range):
        if len(z_range) == 1:
            ptask = []
            # ptask.append(task_iterator(z_range[0]))
            remote_upload(task_iterator(z_range))
        elif len(z_range) > 0:
            ptask = []
            range_list = make_range(z_range, a.threads)
            start = time()

            for irange in range_list:
                ptask.append(task_iterator(irange))
            with ProcessPoolExecutor(max_workers=a.threads) as executor:
                executor.map(remote_upload, ptask)

    status_filename = args.status_output_file
    if status_filename is None:
        status_filename = 'align_block_status_{}.txt'.format(floor(time()))

    def executionLoop(compute_field_z_release, render_z_release=[]):
        with open(status_filename, 'w') as status_file:
            if len(compute_field_z_release) > 0:            
                for z in compute_field_z_release:
                    executeNew(BlockAlignComputeField, compute_field_z_release)
                    z_to_compute_released[z] = True
            if len(render_z_release) > 0:
                executeNew(BlockAlignRender, render_z_release)
                for z in render_z_release:
                    z_to_render_released[z] = True
            with TaskQueue(queue_name=args.completed_queue_name, n_threads=0) as ctq:
                sqs_obj = ctq._api._sqs
                global renders_complete
                while renders_complete < len(block_z_list):
                    msgs = sqs_obj.receive_message(QueueUrl=ctq._api._qurl, MaxNumberOfMessages=10)
                    if 'Messages' not in msgs:
                        sleep(1)
                        continue
                    entriesT = []
                    parsed_msgs = []
                    for i in range(len(msgs['Messages'])):
                        entriesT.append({
                            'ReceiptHandle': msgs['Messages'][i]['ReceiptHandle'],
                            'Id': str(i)
                        })
                        parsed_msgs.append(json.loads(msgs['Messages'][i]['Body']))
                    sqs_obj.delete_message_batch(QueueUrl=ctq._api._qurl, Entries=entriesT)
                    for parsed_msg in parsed_msgs:
                        pos_tuple = (parsed_msg['x'], parsed_msg['y'], parsed_msg['z'])
                        z = pos_tuple[2]
                        if parsed_msg['task'] == 'CF':
                            # import ipdb
                            # ipdb.set_trace()
                            already_processed = chunk_to_compute_processed[pos_tuple]
                            if not already_processed:
                                chunk_to_compute_processed[pos_tuple] = True
                                z_to_computes_processed[z] = z_to_computes_processed[z] + 1
                                if z_to_computes_processed[z] == z_to_number_of_chunks[z]:
                                    if z in z_to_render_released:
                                        if z_to_render_released[z]:
                                            pass
                                            # raise ValueError('Attempt to release render for z={} twice'.format(z+1))
                                        else:
                                            print('CF done for z={}, releasing render for z={}'.format(z, z))
                                            z_to_render_released[z] = True
                                            status_file.write('cf {}'.format(z))
                                            executeNew(BlockAlignRender, [z])
                                elif z_to_computes_processed[z] > z_to_number_of_chunks[z]:
                                    # import ipdb
                                    # ipdb.set_trace()
                                    raise ValueError('More compute chunks processed than exist for z = {}'.format(z))
                        elif parsed_msg['task'] == 'RT':
                            already_processed = chunk_to_render_processed[pos_tuple]
                            if not already_processed:
                                chunk_to_render_processed[pos_tuple] = True
                                z_to_renders_processed[z] = z_to_renders_processed[z] + 1
                                if z_to_renders_processed[z] == z_to_number_of_chunks[z]:
                                    renders_complete = renders_complete + 1
                                    # if renders_complete == 19:
                                        # import ipdb
                                        # ipdb.set_trace()
                                    print('Renders complete: {}'.format(renders_complete))
                                    if z+1 in z_to_compute_released:
                                        if z_to_compute_released[z+1]:
                                            pass
                                            # raise ValueError('Attempt to release compute for z={} twice'.format(z+1))
                                        else:
                                            print('Render done for z={}, releasing cf for z={}'.format(z, z+1))
                                            z_to_compute_released[z+1] = True
                                            status_file.write('rt {}'.format(z))
                                            executeNew(BlockAlignComputeField, [z+1])
                                elif z_to_renders_processed[z] > z_to_number_of_chunks[z]:
                                    raise ValueError('More render chunks processed than exist for z = {}'.format(z))
                        else:
                            raise ValueError('Unsupported task type {}'.format(parsed_msg['task']))

    # # Serial alignment with block stitching
    print("START BLOCK ALIGNMENT")

    # for z_offset in sorted(block_offset_to_z_range.keys()):
    #     z_range = list(block_offset_to_z_range[z_offset])
    #     if do_alignment:
    #         print("ALIGN BLOCK OFFSET {}".format(z_offset))
    #         execute(BlockAlignComputeField, z_range)
    #         if not skip_vv:
    #             print("VECTOR VOTE BLOCK OFFSET {}".format(z_offset))
    #             execute(BlockAlignVectorVote, z_range)
    #     if do_render:
    #         print("RENDER BLOCK OFFSET {}".format(z_offset))
    #         execute(BlockAlignRender, z_range)
    
    if args.recover_status_from_file is None:
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
        executionLoop(first_z_release)
    else:
        first_cf_release, first_rt_release = recover_status_from_file(args.recover_status_from_file)
        # import ipdb
        # ipdb.set_trace()
        executionLoop(first_cf_release, first_rt_release)
        
    print("END BLOCK ALIGNMENT")
    print("START BLOCK STITCHING")
    print("COPY OVERLAPPING IMAGES & FIELDS OF BLOCKS")
    #for z_offset in sorted(stitch_offset_to_z_range.keys()):
    #    z_range = list(stitch_offset_to_z_range[z_offset])
    #    execute(SeethroughStitchRender, z_range=z_range)

    if do_render:
        execute(StitchOverlapCopy, overlap_copy_range)
    for z_offset in sorted(stitch_offset_to_z_range.keys()):
        z_range = list(stitch_offset_to_z_range[z_offset])
        if do_alignment:
            print("ALIGN OVERLAPPING OFFSET {}".format(z_offset))
            execute(StitchAlignComputeField, z_range)
            if not skip_vv:
                print("VECTOR VOTE OVERLAPPING OFFSET {}".format(z_offset))
                execute(StitchAlignVectorVote, z_range)
        if do_render and not skip_vv:
            print("RENDER OVERLAPPING OFFSET {}".format(z_offset))
            execute(StitchAlignRender, z_range)

    if do_alignment and not skip_vv:
        print("COPY OVERLAP ALIGNED FIELDS FOR VECTOR VOTING")
        execute(StitchBroadcastCopy, stitch_range)
        print("VECTOR VOTE STITCHING FIELDS")
        execute(StitchBroadcastVectorVote, block_starts[1:])

    if do_compose:
        execute(StitchCompose, compose_range)
    if do_final_render:
        execute(StitchFinalRender, compose_range)
