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
    parser.add_argument("--final_render_pad", type=int, default=512)
    parser.add_argument(
        "--pad",
        help="the size of the largest displacement expected; should be 2^high_mip",
        type=int,
        default=512,
    )
    parser.add_argument("--block_size", type=int, default=10)
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
        default='float32'
    )
    parser.add_argument(
        "--output_field_dtype",
        type=str,
        help="Data type of output vector fields (typically int16 or float32)",
        default='float32'
    )
    parser.add_argument(
        "--render_dst",
        type=str,
        default=None,
        help="If specified, CloudVolume path to render to instead of default"
    )
    parser.add_argument(
        "--brighten_misalign",
        action='store_true',
        help="If True,brightens misalignments seenthrough"
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
    parser.add_argument('--compose_suffix', type=str, default='', help='string to append to directory names')
    parser.add_argument('--final_render_suffix', type=str, default='', help='string to append to directory names')
    parser.add_argument('--status_output_file', type=str, default=None)
    parser.add_argument('--recover_status_from_file', type=str, default=None)
    parser.add_argument('--block_overlap', type=int, default=1)
    parser.add_argument('--independent_block_file', type=str, default=None)
    parser.add_argument('--write_composing_field', action='store_true')
    parser.add_argument('--generate_params_from_skip_file', type=str, default=None)
    parser.add_argument('--pin_second_starting_section', type=int, default=None)
    parser.add_argument('--write_misalignment_masks', action='store_true')
    parser.add_argument('--write_other_masks', action='store_true')
    parser.add_argument(
        "--write_orig_cv",
        action='store_true',
        help="If True,brightens misalignments seenthrough"
    )
    parser.add_argument('--write_patches', action='store_true')


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

    ind_block_align = args.independent_block_file is not None

    block_size = args.block_size
    do_alignment = not args.skip_alignment
    do_stitching = not args.skip_stitching and not ind_block_align
    do_compose = not args.skip_compose and not ind_block_align
    do_final_render = not args.skip_final_render and not ind_block_align
    blackout_op = args.blackout_op
    stitch_suffix = args.stitch_suffix
    output_field_dtype = args.output_field_dtype
    write_composing_field = args.write_composing_field
    write_misalignment_masks = args.write_misalignment_masks
    write_other_masks = args.write_other_masks
    write_orig_cv = args.write_orig_cv
    write_patches = args.write_patches

    if write_misalignment_masks:
        # Need composing field to produce misalignment masks
        # assert(write_composing_field)
        write_composing_field = True

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

    mask_dict = {}
    for mask in src_masks:
        mask_mip = mask.dst_mip or mip
        if mask_mip in mask_dict:
            mask_dict[mask_mip].append(mask)
        else:
            mask_dict[mask_mip] = [mask]

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




    render_dst = args.dst_path
    if args.render_dst is not None:
        render_dst = args.render_dst

    # Create dst CloudVolumes for odd & even blocks, since blocks overlap by tgt_radius
    block_dsts = {}
    misalignment_dsts = {}
    block_types = ["even", "odd"]
    misalignment_block_types = ["misalignment_even", "misalignment_odd"]
    for i, block_type in enumerate(block_types):
        block_dst = cm.create(
            join(render_dst, "image_blocks", block_type),
            data_type=args.img_dtype,
            num_channels=1,
            fill_missing=True,
            overwrite=do_alignment,
        )
        block_dsts[i] = block_dst.path
    for i, block_type in enumerate(misalignment_block_types):
        misalignment_block_dst = cm.create(
            join(render_dst, "image_blocks", block_type),
            data_type='uint8',
            num_channels=1,
            fill_missing=True,
            overwrite=do_alignment,
        )
        misalignment_dsts[i] = misalignment_block_dst.path
    if write_patches:
        src_patch_dsts = {}
        src_patch_block_types = ["src_patch_even", "src_patch_odd"]
        for i, block_type in enumerate(src_patch_block_types):
            src_patch_dst = cm.create(
                join(render_dst, "image_blocks", block_type),
                data_type=args.img_dtype,
                num_channels=1,
                fill_missing=True,
                overwrite=do_alignment,
            )
            src_patch_dsts[i] = src_patch_dst.path

        tgt_patch_dsts = {}
        tgt_patch_block_types = ["tgt_patch_even", "tgt_patch_odd"]
        for i, block_type in enumerate(tgt_patch_block_types):
            tgt_patch_dst = cm.create(
                join(render_dst, "image_blocks", block_type),
                data_type=args.img_dtype,
                num_channels=1,
                fill_missing=True,
                overwrite=do_alignment,
            )
            tgt_patch_dsts[i] = tgt_patch_dst.path

    # Compile bbox, model, vvote_offsets for each z index, along with indices to skip
    bbox_lookup = {}
    model_lookup = {}
    tgt_radius_lookup = {}
    vvote_lookup = {}
    alignment_z_starts = [args.z_start]
    last_alignment_start = args.z_start
    minimum_block_size = 5
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
                bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
                for z in range(z_start, z_stop):
                    bbox_lookup[z] = bbox
                    model_lookup[z] = model_path
                    tgt_radius_lookup[z] = tgt_radius
                    vvote_lookup[z] = [-i for i in range(1, tgt_radius + 1)]
                if args.generate_params_from_skip_file is not None:
                    continue
                while z_start - last_alignment_start > (block_size + minimum_block_size):
                    last_alignment_start = last_alignment_start + block_size
                    alignment_z_starts.append(last_alignment_start)
                if z_start > last_alignment_start:
                    last_alignment_start = z_start
                    alignment_z_starts.append(z_start)

    if args.generate_params_from_skip_file is None:
        while min(z_stop, args.z_stop) - last_alignment_start > block_size:
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
        if args.pin_second_starting_section is not None:
            cur_start = args.pin_second_starting_section
            alignment_z_starts.append(cur_start)
        while cur_start < args.z_stop:
            # alignment_z_starts.append(cur_start)
            cur_stop = cur_start + block_size
            prev_stop = cur_stop
            while cur_stop in skip_sections:
                cur_stop = cur_stop + 1
            if cur_stop < args.z_stop:
                alignment_z_starts.append(cur_stop)
            cur_start = prev_stop
            

    # Filter out skipped sections from vvote_offsets
    min_offset = 0
    for z, tgt_radius in vvote_lookup.items():
        offset = 0
        for i, r in enumerate(tgt_radius):
            tgt_radius[i] = r + offset
        min_offset = min(min_offset, r + offset)
        offset = 0
        vvote_lookup[z] = tgt_radius

    # Adjust block starts so they don't start on a skipped section
    if ind_block_align:
        block_starts = []
        initial_block_starts = block_starts
        block_stops = []
        with open(args.independent_block_file, 'r') as f:
            line = f.readline()
            while line:
                block_starts.append(int(line))
                block_stops.append(int(line) + block_size)
                line = f.readline()
    else:
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
    misalignment_dst_lookup = {}
    src_patch_lookup = {}
    tgt_patch_lookup = {}
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
                if write_patches:
                    src_patch_lookup[z] = src_patch_dsts[even_odd]
                    tgt_patch_lookup[z] = tgt_patch_dsts[even_odd]
                misalignment_dst_lookup[z] = misalignment_dsts[even_odd]
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
    starter_restart = 0
    copy_offset_to_z_range = {
        k: v for k, v in copy_offset_to_z_range.items() if k == 0
    }
    starter_offset_to_z_range = {
        k: v for k, v in starter_offset_to_z_range.items() if k <= starter_restart
    }
    block_offset_to_z_range = {
        k: v for k, v in block_offset_to_z_range.items() if k >= 0
    }
    copy_range = [z for z_range in copy_offset_to_z_range.values() for z in z_range]
    starter_range = [
        z for z_range in starter_offset_to_z_range.values() for z in z_range
    ]
    overlap_copy_range = list(overlap_copy_range)
    for i in range(len(overlap_copy_range)):
        overlap_copy_range[i] = overlap_copy_range[i] + args.block_overlap - 1

    # Determine the number of sections needed to stitch (no stitching for block 0)
    stitch_offset_to_z_range = {i: [] for i in range(1, block_size + 1)}
    block_start_to_stitch_offsets = {i: [] for i in block_starts[1:]}
    for bs, be in zip(block_starts[1:], block_stops[1:]):
        max_offset = 0
        for i, z in enumerate(range(bs, be + 1)):
            if i > 0:
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


    compose_range = range(args.z_start, args.z_stop)
    render_range = range(args.z_start+1, args.z_stop)
    if do_compose:
        decay_dist = args.decay_dist
        influencing_blocks_lookup = {z: [] for z in compose_range}
        for b_start in block_starts:
            b_stitch = b_start + args.block_overlap
            for z in range(b_stitch, b_stitch+decay_dist+1):
              if z < args.z_stop:
                  influencing_blocks_lookup[z].append(b_stitch)



    # Create field CloudVolumes
    print("Creating field & overlap CloudVolumes")
    block_pair_fields = {}
    for z_offset in offset_range:
        block_pair_fields[z_offset] = cm.create(
            join(args.dst_path, "field", "block", str(z_offset)),
            data_type=output_field_dtype,
            num_channels=2,
            fill_missing=True,
            overwrite=do_alignment,
        ).path
    block_vvote_field = cm.create(
        join(args.dst_path, "field", "vvote"),
        data_type=output_field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=do_alignment,
    ).path
    block_overlap_field = cm.create(
        join(args.dst_path, "field", "overlap"),
        data_type=output_field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=do_alignment,
    ).path
    if write_misalignment_masks:
        misalignment_mask_cv = cm.create(
            join(args.dst_path, "image_blocks", "misalignment_mask"),
            data_type='uint8',
            num_channels=1,
            fill_missing=True,
            overwrite=do_alignment,
        ).path
        misalignment_mask_overlap_cv = cm.create(
            join(args.dst_path, "image_blocks", "misalignment_mask_overlap"),
            data_type='uint8',
            num_channels=1,
            fill_missing=True,
            overwrite=do_alignment,
        ).path
    if write_orig_cv:
        orig_cv = cm.create(
            join(args.dst_path, "image_blocks", "no_st"),
            data_type=args.img_dtype,
            num_channels=1,
            fill_missing=True,
            overwrite=do_alignment
        ).path
        orig_overlap_cv = cm.create(
            join(args.dst_path, "image_blocks", "no_st_overlap"),
            data_type=args.img_dtype,
            num_channels=1,
            fill_missing=True,
            overwrite=do_alignment
        ).path
    stitch_pair_fields = {}
    for z_offset in offset_range:
        stitch_pair_fields[z_offset] = cm.create(
            join(args.dst_path, "field", "stitch{}".format(args.stitch_suffix), str(z_offset)),
            data_type=output_field_dtype,
            num_channels=2,
            fill_missing=True,
            overwrite=do_stitching,
        ).path
    overlap_vvote_field = cm.create(
        join(args.dst_path, "field", "stitch{}".format(args.stitch_suffix), "vvote", "field"),
        data_type=output_field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=do_stitching,
    ).path
    overlap_image = cm.create(
        join(args.dst_path, "field", "stitch{}".format(args.stitch_suffix), "vvote", "image"),
        data_type=args.img_dtype,
        num_channels=1,
        fill_missing=True,
        overwrite=do_stitching,
    ).path
    stitch_fields = {}
    for z_offset in offset_range:
        stitch_fields[z_offset] = cm.create(
            join(args.dst_path, "field", "stitch{}".format(args.stitch_suffix), "vvote", str(z_offset)),
            data_type=output_field_dtype,
            num_channels=2,
            fill_missing=True,
            overwrite=do_stitching,
        ).path
    broadcasting_field = cm.create(
        join(args.dst_path, "field", "stitch{}".format(args.stitch_suffix), "broadcasting"),
        data_type=output_field_dtype,
        num_channels=2,
        fill_missing=True,
        overwrite=do_stitching,
    ).path

    compose_field = cm.create(join(args.dst_path, 'field',
                        'stitch{}'.format(args.stitch_suffix), 'compose{}'.format(args.compose_suffix)),
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
        final_dst = cmr.create(join(args.dst_path, 'image_stitch{}{}{}'.format(args.stitch_suffix, args.compose_suffix, args.final_render_suffix)),
                            data_type='uint8', num_channels=1, fill_missing=True,
                            overwrite=do_final_render).path
        
        if write_misalignment_masks:
            final_misalignment_masks = cm.create(join(args.dst_path, 'misalignment_stitch{}{}{}'.format(args.stitch_suffix, args.compose_suffix, args.final_render_suffix)),
                            data_type='uint8', num_channels=1, fill_missing=True, overwrite=True).path
        if write_other_masks:
            mask_cv_dict = {}
            for final_mask_mip in mask_dict:
                final_mask_mip_cv = cm.create(join(args.dst_path, 'mask_mip{}_stitch{}'.format(final_mask_mip, args.final_render_suffix)),
                            data_type='uint8', num_channels=1, fill_missing=True, overwrite=True).path
                mask_cv_dict[final_mask_mip] = final_mask_mip_cv

    if write_composing_field:
        composing_field = cm.create(join(args.dst_path, 'field',
                        'stitch{}'.format(args.stitch_suffix), 'compose_diff'),
                        data_type=output_field_dtype, num_channels=2,
                        fill_missing=True, overwrite=do_compose).path

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
                    seethrough=False,
                    seethrough_misalign=False
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


    class BlockAlignComputeField(object):
        def __init__(self, z_range, block_starts_for_range):
            self.z_range = z_range
            self.block_starts = block_starts_for_range

        def __iter__(self):
            for i in range(len(self.z_range)):
                src_z = self.z_range[i]
                block_start = self.block_starts[i]
                dst = block_dst_lookup[self.block_starts[i]+1]
                bbox = bbox_lookup[src_z]
                model_path = model_lookup[src_z]
                tgt_offsets = vvote_lookup[src_z]
                for tgt_offset in tgt_offsets:
                    tgt_z = src_z + tgt_offset
                    fine_field = block_vvote_field
                    if self.block_starts[i] != block_start_lookup[src_z]:
                        fine_field = block_overlap_field
                    if tgt_z == self.block_starts[i]:
                        tgt_field = block_pair_fields[0]
                    elif block_start_lookup[tgt_z] != self.block_starts[i]:
                        tgt_field = block_overlap_field
                    else:
                        tgt_field = block_vvote_field
                    write_src_patch_cv = None
                    write_tgt_patch_cv = None
                    if write_patches:
                        write_src_patch_cv = src_patch_lookup[self.block_starts[i]+1]
                        write_tgt_patch_cv = tgt_patch_lookup[self.block_starts[i]+1]
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
                        report=True,
                        block_start=block_start,
                        write_src_patch_cv=write_src_patch_cv,
                        write_tgt_patch_cv=write_tgt_patch_cv
                    )
                    yield from t

    class BlockAlignRender(object):
        def __init__(self, z_range, block_starts_for_range):
            self.z_range = z_range
            self.block_starts = block_starts_for_range

        def __iter__(self):
            for i in range(len(self.z_range)):
                z = self.z_range[i]
                block_start = self.block_starts[i]
                dst = block_dst_lookup[self.block_starts[i]+1]
                misalignment_count_cv = misalignment_dst_lookup[self.block_starts[i]+1]
                bbox = bbox_lookup[z]
                misalignment_mask_cv_to_use = None
                orig_image_cv = None
                if block_start_lookup[z] != self.block_starts[i]:
                    field = block_overlap_field
                    if write_misalignment_masks:
                        misalignment_mask_cv_to_use = misalignment_mask_overlap_cv
                    if write_orig_cv:
                        orig_image_cv = orig_overlap_cv
                else:
                    field = block_vvote_field
                    if write_misalignment_masks:
                        misalignment_mask_cv_to_use = misalignment_mask_cv
                    if write_orig_cv:
                        orig_image_cv = orig_cv                                  
                t = a.render(
                    cm,
                    src,
                    field,
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
                    brighten_misalign=args.brighten_misalign,
                    report=True,
                    block_start=block_start,
                    misalignment_mask_cv=misalignment_mask_cv_to_use,
                    orig_image_cv=orig_image_cv,
                    misalignment_count_cv=misalignment_count_cv
                )
                yield from t


    class StitchOverlapCopy:
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                dst = block_dst_lookup[z-args.block_overlap]
                bbox = bbox_lookup[z]
                ti = a.copy(cm, dst, overlap_image, z, z, bbox, mip, is_field=False)
                yield from ti

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
                    field = broadcasting_field
                    t = a.compute_field(cm, model_path, block_dst, overlap_image, field,
                                        z, tgt_z, bbox, mip, pad,
                                        src_masks=src_masks,
                                        tgt_masks=tgt_masks,
                                        # prev_field_cv=None,
                                        prev_field_cv=block_overlap_field,
                                        cur_field_cv=block_vvote_field,
                                        coarse_field_cv=coarse_field_cv,
                                        coarse_field_mip=coarse_field_mip,
                                        prev_field_z=tgt_z,stitch=True,unaligned_cv=src)
                    yield from t

    class StitchPreCompose(object):
        def __init__(self, z_range):
          self.z_range = z_range

        def __iter__(self):
          for z in self.z_range:
            influencing_blocks = influencing_blocks_lookup[z]
            factors = [interpolate(z, bs, decay_dist) for bs in influencing_blocks]
            print('z={}\ninfluencing_blocks {}\nfactors {}'.format(z, influencing_blocks,
                                                                   factors))
            bbox = bbox_lookup[z]
            cv_list = [broadcasting_field] * len(influencing_blocks)
            z_list = list(influencing_blocks)
            if len(cv_list) > 0: 
                t = a.multi_compose(cm, cv_list, composing_field, z_list, z, bbox,
                                mip, mip, factors, pad)
                yield from t
    
    class StitchCompose(object):
        def __init__(self, z_range):
          self.z_range = z_range

        def __iter__(self):
          for z in self.z_range:
            if z == args.z_start:
                bbox = bbox_lookup[z]
                t = a.cloud_upsample_field(
                    cm,
                    coarse_field_cv,
                    compose_field,
                    src_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=coarse_field_mip,
                    dst_mip=mip
                )
                yield from t
            else:
                influencing_blocks = influencing_blocks_lookup[z]
                factors = [interpolate(z, bs, decay_dist) for bs in influencing_blocks]
                factors += [1.]
                print('z={}\ninfluencing_blocks {}\nfactors {}'.format(z, influencing_blocks,
                                                                    factors))
                bbox = bbox_lookup[z]
                field = block_vvote_field
                if z in block_start_lookup:
                    z_block_start = block_start_lookup[z]
                    if (z - args.block_overlap + 1) in block_start_lookup and block_start_lookup[z - args.block_overlap + 1] != z_block_start:
                        field = block_overlap_field
                if write_composing_field:
                    t = a.multi_compose(cm, [composing_field, field], compose_field, [z, z], z, bbox,
                                    mip, mip, [1., 1.], pad)
                    yield from t
                else:
                    cv_list = [broadcasting_field]*len(influencing_blocks) + [field]
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
            field = compose_field
            field_mip = mip
            t = a.render(cmr, src, field, final_dst, src_z=z,
                         field_z=z, dst_z=z, bbox=bbox,
                         src_mip=final_render_mip, field_mip=field_mip, pad=final_render_pad,
                         masks=[], blackout_op=blackout_op)
            tasks = t
            if write_misalignment_masks:
                misalignment_mask_cv_to_use = misalignment_mask_cv
                if z in block_start_lookup:
                    z_block_start = block_start_lookup[z]
                    if (z - args.block_overlap + 1) in block_start_lookup and block_start_lookup[z - args.block_overlap + 1] != z_block_start:
                        misalignment_mask_cv_to_use = misalignment_mask_overlap_cv
                t_mask = a.render(cmr, misalignment_mask_cv_to_use, composing_field, final_misalignment_masks, src_z=z,
                            field_z=z, dst_z=z, bbox=bbox,
                            src_mip=mip, field_mip=mip, pad=final_render_pad)
                tasks = tasks + t_mask
            if write_other_masks:
                if len(src_masks) > 0:
                    for cur_mask_cv_mip in mask_cv_dict:
                        cur_mask_cv = mask_cv_dict[cur_mask_cv_mip]
                        t_other_masks = a.render_masks(cm, cur_mask_cv, field, src_z=z,
                                field_z=z, dst_z=z, bbox=bbox,
                                dst_mip=cur_mask_cv_mip, pad=args.pad,
                                masks=mask_dict[cur_mask_cv], blackout_op='none')
                        tasks = tasks + t_other_masks
            yield from tasks


    def break_into_chunks(chunk_size, offset, mip, z_list, max_mip=12):
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
        for z in z_list:
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
            number_of_chunks_in_z = x_size * y_size

            for xs in range(calign_x_range[0], calign_x_range[1], chunk_size[0]):
                for ys in range(calign_y_range[0], calign_y_range[1], chunk_size[1]):
                    chunks.append((xs, ys, int(z)))
        return chunks, z_to_number_of_chunks, number_of_chunks_in_z

    block_z_to_compute_released = {}
    block_z_to_render_released = {}
    block_z_to_computes_processed = {}
    block_z_to_renders_processed = {}
    block_z_to_last_compute_time = {}
    block_z_to_last_render_time = {}
    block_chunk_to_compute_processed = {}
    block_chunk_to_render_processed = {}

    total_sections_to_align = 0
    total_sections_aligned = 0
    blocks_finished = 0

    if ind_block_align:
        for i in range(len(block_starts)):
            cur_bs = block_starts[i]
            end_bs = cur_bs + block_size
            zs_for_cur_block = [*range(cur_bs+1, end_bs)]
            total_sections_to_align = total_sections_to_align + len(zs_for_cur_block)
            block_z_to_compute_released[cur_bs] = dict(zip(zs_for_cur_block, [False] * len(zs_for_cur_block)))
            block_z_to_render_released[cur_bs] = dict(zip(zs_for_cur_block, [False] * len(zs_for_cur_block)))
            block_z_to_computes_processed[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            block_z_to_renders_processed[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            block_z_to_last_compute_time[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            block_z_to_last_render_time[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            chunks, z_to_number_of_chunks, number_of_chunks_in_z = break_into_chunks(cm.dst_chunk_sizes[mip], cm.dst_voxel_offsets[mip], mip=mip, z_list=zs_for_cur_block, max_mip=cm.max_mip)
            block_chunk_to_compute_processed[cur_bs] = dict(zip(chunks, [False] * len(chunks)))
            block_chunk_to_render_processed[cur_bs] = dict(zip(chunks, [False] * len(chunks)))
    else:
        for i in range(len(initial_block_starts)-1):
            cur_bs = initial_block_starts[i]
            end_bs = min(initial_block_starts[-1], initial_block_starts[i+1] + args.block_overlap)
            zs_for_cur_block = [*range(cur_bs+1, end_bs)]
            total_sections_to_align = total_sections_to_align + len(zs_for_cur_block)
            block_z_to_compute_released[cur_bs] = dict(zip(zs_for_cur_block, [False] * len(zs_for_cur_block)))
            block_z_to_render_released[cur_bs] = dict(zip(zs_for_cur_block, [False] * len(zs_for_cur_block)))
            block_z_to_computes_processed[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            block_z_to_renders_processed[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            block_z_to_last_compute_time[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            block_z_to_last_render_time[cur_bs] = dict(zip(zs_for_cur_block, [0] * len(zs_for_cur_block)))
            chunks, z_to_number_of_chunks, number_of_chunks_in_z = break_into_chunks(cm.dst_chunk_sizes[mip], cm.dst_voxel_offsets[mip], mip=mip, z_list=zs_for_cur_block, max_mip=cm.max_mip)
            block_chunk_to_compute_processed[cur_bs] = dict(zip(chunks, [False] * len(chunks)))
            block_chunk_to_render_processed[cur_bs] = dict(zip(chunks, [False] * len(chunks)))

    def recover_status_from_file(filename):
        global total_sections_aligned
        with open(filename, 'r') as recover_file:
            line = recover_file.readline()
            while line:
                spl = line.split()
                bs = int(spl[1])
                task = spl[2]
                z = int(spl[3])
                if task == 'cf':
                    block_z_to_compute_released[bs][z] = True
                    block_z_to_computes_processed[bs][z] = number_of_chunks_in_z
                elif task == 'rt':
                    if not block_z_to_render_released[bs][z]:
                        total_sections_aligned = total_sections_aligned + 1
                    block_z_to_render_released[bs][z] = True
                    block_z_to_renders_processed[bs][z] = number_of_chunks_in_z
                line = recover_file.readline()

    def generate_first_releases():
        new_cf_list = []
        new_rt_list = []
        cf_block_start = []
        rt_block_start = []
        for bs in initial_block_starts[:-1]:
            z = bs + 1
            while z in block_z_to_compute_released[bs]:
                if not block_z_to_compute_released[bs][z]:
                    new_cf_list.append(z)
                    cf_block_start.append(bs)
                    break
                if not block_z_to_render_released[bs][z]:
                    new_rt_list.append(z)
                    rt_block_start.append(bs)
                    break
                z = z + 1
        return new_cf_list, new_rt_list, cf_block_start, rt_block_start

    def executeNew(task_iterator, z_range, respective_block_starts):
        assert len(z_range) == len(respective_block_starts)
        if len(z_range) == 1:
            ptask = []
            remote_upload(task_iterator(z_range, respective_block_starts))
        elif len(z_range) > 0:
            ptask = []
            range_list = make_range(z_range, a.threads)
            block_range_list = make_range(respective_block_starts, a.threads)
            start = time()

            for i in range(len(range_list)):
                irange = range_list[i]
                iblock_starts = block_range_list[i]
                ptask.append(task_iterator(irange, iblock_starts))
            with ProcessPoolExecutor(max_workers=a.threads) as executor:
                executor.map(remote_upload, ptask)

    status_filename = args.status_output_file
    if status_filename is None:
        status_filename = 'align_block_status_{}.txt'.format(floor(time()))

    profile_filename = 'profile_align_blocks_{}.txt'.format(floor(time()))
    retry_filename = 'retry_align_blocks_{}.txt'.format(floor(time()))
    profile_file = open(profile_filename, 'w')
    retry_file = open(retry_filename, 'w')
    receive_time = 0
    process_time = 0
    delete_time = 0

    def release_compute_and_render(compute_field_z_release, render_z_release, cf_block_starts, rt_block_starts):
        assert len(compute_field_z_release) == len(cf_block_starts)
        assert len(render_z_release) == len(rt_block_starts)
        if len(compute_field_z_release) > 0:
            executeNew(BlockAlignComputeField, compute_field_z_release, cf_block_starts)
            for i in range(len(compute_field_z_release)):
                block_start = cf_block_starts[i]
                z = compute_field_z_release[i]
                block_z_to_compute_released[block_start][z] = True
                block_z_to_last_compute_time[block_start][z] = time()
        if len(render_z_release) > 0:
            executeNew(BlockAlignRender, render_z_release, rt_block_starts)
            for i in range(len(render_z_release)):
                block_start = rt_block_starts[i]
                z = render_z_release[i]
                block_z_to_render_released[block_start][z] = True
                block_z_to_last_render_time[block_start][z] = time()

    poll_time = 300

    def get_lagged_tasks(resend_time):
        cf_list = []
        rt_list = []
        cf_block_start = []
        rt_block_start = []
        time_check = time()
        for block_start in block_z_to_compute_released:
            zs_for_block = block_z_to_compute_released[block_start]
            for z in zs_for_block:
                if block_z_to_compute_released[block_start][z]:
                    if block_z_to_computes_processed[block_start][z] != number_of_chunks_in_z:
                        if time_check - block_z_to_last_compute_time[block_start][z] >= resend_time:
                            cf_list.append(z)
                            cf_block_start.append(block_start)
                else:
                    break
        for block_start in block_z_to_render_released:
            zs_for_block = block_z_to_render_released[block_start]
            for z in zs_for_block:
                if block_z_to_render_released[block_start][z]:
                    if block_z_to_renders_processed[block_start][z] != number_of_chunks_in_z:
                        if time_check - block_z_to_last_render_time[block_start][z] >= resend_time:
                            rt_list.append(z)
                            rt_block_start.append(block_start)
                else:
                    break
        return cf_list, rt_list, cf_block_start, rt_block_start

    def executionLoop(compute_field_z_release, render_z_release, cf_block_starts, rt_block_starts):
        release_compute_and_render(compute_field_z_release, render_z_release, cf_block_starts, rt_block_starts)
        first_poll_time = time()
        with open(status_filename, 'w') as status_file:
            with TaskQueue(queue_name=args.completed_queue_name, n_threads=0) as ctq:
                sqs_obj = ctq._api._sqs
                global total_sections_aligned
                global renders_complete
                global receive_time
                global process_time
                global delete_time
                global blocks_finished
                empty_in_a_row = 0
                while total_sections_aligned < total_sections_to_align:
                    check_poll_time = time()
                    if check_poll_time - first_poll_time >= poll_time:
                        first_poll_time = check_poll_time
                        cf_list, rt_list, cf_block_start, rt_block_start = get_lagged_tasks(72000)
                        if len(cf_list) > 0 or len(rt_list) > 0:
                            for cf_i in range(len(cf_list)):
                                retry_file.write('Timed out: bs {} cf {} time {}\n'.format(cf_block_start[cf_i], cf_list[cf_i], check_poll_time))
                            for rt_i in range(len(rt_list)):
                                retry_file.write('Timed out: bs {} rt {} time {}\n'.format(rt_block_start[rt_i], rt_list[rt_i], check_poll_time))
                            print('Restarting tasks because too much time has passed\n')
                        release_compute_and_render(cf_list, rt_list, cf_block_start, rt_block_start)
                    before_receive_time = time()
                    msgs = sqs_obj.receive_message(QueueUrl=ctq._api._qurl, MaxNumberOfMessages=10)
                    receive_time = receive_time + time() - before_receive_time
                    if 'Messages' not in msgs:
                        empty_in_a_row = empty_in_a_row + 1
                        if empty_in_a_row >= 20:
                            if a.sqs_is_empty_fast():
                                cf_list, rt_list, cf_block_start, rt_block_start = get_lagged_tasks(0)
                                for cf_i in range(len(cf_list)):
                                    retry_file.write('Queue empty: bs {} cf {} time {}\n'.format(cf_block_start[cf_i], cf_list[cf_i], check_poll_time))
                                for rt_i in range(len(rt_list)):
                                    retry_file.write('Queue empty: bs {} rt {} time {}\n'.format(rt_block_start[rt_i], rt_list[rt_i], check_poll_time))
                                print('Restarting tasks because queue is empty\n')
                                release_compute_and_render(cf_list, rt_list, cf_block_start, rt_block_start)
                        else:
                            sleep(1)
                        continue
                    empty_in_a_row = 0
                    entriesT = []
                    parsed_msgs = []
                    for i in range(len(msgs['Messages'])):
                        entriesT.append({
                            'ReceiptHandle': msgs['Messages'][i]['ReceiptHandle'],
                            'Id': str(i)
                        })
                        parsed_msgs.append(json.loads(msgs['Messages'][i]['Body']))
                    before_delete_time = time()
                    sqs_obj.delete_message_batch(QueueUrl=ctq._api._qurl, Entries=entriesT)
                    delete_time = delete_time + time() - before_delete_time
                    before_process_time = time()
                    for parsed_msg in parsed_msgs:
                        pos_tuple = (parsed_msg['x'], parsed_msg['y'], parsed_msg['z'])
                        z = pos_tuple[2]
                        block_start = parsed_msg['block_start']
                        if parsed_msg['task'] == 'CF':
                            already_processed = block_chunk_to_compute_processed[block_start][pos_tuple]
                            if not already_processed:
                                block_z_to_last_compute_time[block_start][z] = time()
                                block_chunk_to_compute_processed[block_start][pos_tuple] = True
                                block_z_to_computes_processed[block_start][z] = block_z_to_computes_processed[block_start][z] + 1
                                if block_z_to_computes_processed[block_start][z] == number_of_chunks_in_z:
                                    if z in block_z_to_render_released[block_start]:
                                        if block_z_to_render_released[block_start][z]:
                                            pass
                                            # raise ValueError('Attempt to release render for z={} twice'.format(z+1))
                                        else:
                                            print('CF done for z={}, releasing render for z={}'.format(z, z))
                                            block_z_to_render_released[block_start][z] = True
                                            status_file.write('bs {} cf {}\n'.format(block_start, z))
                                            profile_file.write('process time {}\n'.format(process_time))
                                            profile_file.write('receive time {}\n'.format(receive_time))
                                            profile_file.write('delete time {}\n'.format(delete_time))
                                            block_z_to_last_render_time[block_start][z] = time()
                                            executeNew(BlockAlignRender, [z], [block_start])
                                elif block_z_to_computes_processed[block_start][z] > number_of_chunks_in_z:
                                    raise ValueError('More compute chunks processed than exist for z = {}'.format(z))
                        elif parsed_msg['task'] == 'RT':
                            already_processed = block_chunk_to_render_processed[block_start][pos_tuple]
                            if not already_processed:
                                block_z_to_last_render_time[block_start][z] = time()
                                block_chunk_to_render_processed[block_start][pos_tuple] = True
                                block_z_to_renders_processed[block_start][z] = block_z_to_renders_processed[block_start][z] + 1
                                if block_z_to_renders_processed[block_start][z] == number_of_chunks_in_z:
                                    total_sections_aligned = total_sections_aligned + 1
                                    print('Renders complete: {}'.format(total_sections_aligned))
                                    status_file.write('bs {} rt {}\n'.format(block_start, z))
                                    if z+1 in block_z_to_compute_released[block_start]:
                                        if block_z_to_compute_released[block_start][z+1]:
                                            pass
                                            # raise ValueError('Attempt to release compute for z={} twice'.format(z+1))
                                        else:
                                            print('Render done for z={}, releasing cf for z={}'.format(z, z+1))
                                            block_z_to_compute_released[block_start][z+1] = True
                                            profile_file.write('process time {}\n'.format(process_time))
                                            profile_file.write('receive time {}\n'.format(receive_time))
                                            profile_file.write('delete time {}\n'.format(delete_time))
                                            block_z_to_last_compute_time[block_start][z+1] = time()
                                            executeNew(BlockAlignComputeField, [z+1], [block_start])
                                    else:
                                        blocks_finished = blocks_finished + 1
                                        print('Blocks finished: {}'.format(blocks_finished))
                                elif block_z_to_renders_processed[block_start][z] > number_of_chunks_in_z:
                                    raise ValueError('More render chunks processed than exist for z = {}'.format(z))
                        else:
                            raise ValueError('Unsupported task type {}'.format(parsed_msg['task']))
                    process_time = process_time + time() - before_process_time

    # # Serial alignment with block stitching
    print("START BLOCK ALIGNMENT")

    if args.recover_status_from_file is None:
        if do_alignment:
            print("COPY STARTING SECTION OF ALL BLOCKS")
            execute(StarterCopy, copy_range)
            if coarse_field_cv is not None:
                print("UPSAMPLE STARTING SECTION COARSE FIELDS OF ALL BLOCKS")
                # execute(StarterUpsampleField, copy_range)
    else:
        recover_status_from_file(args.recover_status_from_file)

    if a.distributed:
        if do_alignment:
            cf_list, rt_list, cf_block_start, rt_block_start = generate_first_releases()
            executionLoop(cf_list, rt_list, cf_block_start, rt_block_start)
    else:
        for z_offset in sorted(block_offset_to_z_range.keys()):
            z_range = list(block_offset_to_z_range[z_offset])
            if do_alignment:
                print("ALIGN BLOCK OFFSET {}".format(z_offset))
                execute(BlockAlignComputeField, z_range)
            if do_alignment:
                print("RENDER BLOCK OFFSET {}".format(z_offset))
                execute(BlockAlignRender, z_range)

    print("END BLOCK ALIGNMENT")
    print("START BLOCK STITCHING")
    print("COPY OVERLAPPING IMAGES & FIELDS OF BLOCKS")

    if do_stitching:
        execute(StitchOverlapCopy, overlap_copy_range)
    for z_offset in sorted(stitch_offset_to_z_range.keys()):
        z_range = list(stitch_offset_to_z_range[z_offset])
        for i in range(len(z_range)):
            z_range[i] = z_range[i] + args.block_overlap - 1
        if do_stitching:
            print("ALIGN OVERLAPPING OFFSET {}".format(z_offset))
            execute(StitchAlignComputeField, z_range)

    # compose_range = range(17,400)

    if do_compose:
        if write_misalignment_masks:
            print("COMPOSING FOR MISALIGNMENT MASKS")
            execute(StitchPreCompose, compose_range)
        print("FINAL COMPOSING")
        execute(StitchCompose, compose_range)
    if do_final_render:
        print("FINAL RENDERING")
        execute(StitchFinalRender, compose_range)
