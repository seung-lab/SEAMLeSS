from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from os.path import join
from time import time

from taskqueue import GreenTaskQueue, LocalTaskQueue, MockTaskQueue

def get_starter_copy_task(environment):
    class StarterCopy:
        def __init__(self, z_range):
            print(z_range)
            self.z_range = z_range

        def __iter__(self):
            for z in self.z_range:
                block_dst = environment['starter_dst_lookup'][z]
                bbox = environment['bbox_lookup'][z]
                # t =  a.copy(cm, src, block_dst, z, z, bbox, mip, is_field=False,
                #             mask_cv=src_mask_cv, mask_mip=src_mask_mip, mask_val=src_mask_val)

                t = a.render(
                    environment['cm'],
                    environment['src'],
                    environment['coarse_field_cv'],
                    block_dst,
                    src_z=z,
                    field_z=z,
                    dst_z=z,
                    bbox=bbox,
                    src_mip=environment['render_mip'],
                    field_mip=environment['coarse_field_mip'],
                    mask_cv=environment['src_mask_cv'],
                    mask_val=environment['src_mask_val'],
                    mask_mip=environment['src_mask_mip'],
                )
                yield from t
    return StarterCopy

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

class TaskExecutor:
    def __init__(self, aligner, queue_name, do_alignment=True,
                 dry_run=False):
        self.aligner = aligner
        self.queue_name = queue_name
        self.do_alignment = do_alignment
        self.dry_run = dry_run

# Task scheduling functions
    def remote_upload(self, tasks):
        with GreenTaskQueue(queue_name=self.queue_name) as tq:
            tq.insert_all(tasks)

    def execute(self, task_iterator, z_range):
        if len(z_range) > 0:
            ptask = []
            range_list = make_range(z_range, self.aligner.threads)
            start = time()

            for irange in range_list:
                ptask.append(task_iterator(irange))
            if self.dry_run:
                for t in ptask:
                    tq = MockTaskQueue(parallel=1)
                    tq.insert_all(t, args=[self.aligner])
            else:
                if self.aligner.distributed:
                    with ProcessPoolExecutor(max_workers=self.aligner.threads) as executor:
                        executor.map(self.remote_upload, ptask)
                else:
                    for t in ptask:
                        tq = LocalTaskQueue(parallel=1)
                        tq.insert_all(t, args=[self.aligner])

            end = time()
            diff = end - start
            print("Sending {} use time: {}".format(task_iterator, diff))
            if self.aligner.distributed:
                print("Run {}".format(task_iterator))
                # wait
                start = time()
                if self.do_alignment:
                    self.aligner.wait_for_sqs_empty()
                end = time()
                diff = end - start
                print("Executing {} use time: {}\n".format(task_iterator, diff))



