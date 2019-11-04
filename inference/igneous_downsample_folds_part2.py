import sys
from taskqueue import TaskQueue
import igneous.task_creation as tc
from cloudvolume import Bbox

with TaskQueue('deepalign-igneous-1') as tq:
    tasks = tc.create_downsampling_tasks(
        'precomputed://gs://seunglab_minnie_phase3/alignment/precoarse_vv5_tempdiv4_step128_maxdisp128/warped_folds_final',
        mip=7,
        num_mips=1,
        fill_missing=True,
        preserve_chunk_size=True,
        chunk_size=[512, 512, 1],
        bounds=Bbox((-20992, -20992, 14000), (503296, 372224, 28000)),
        sparse=True,
        delete_black_uploads=True,
    )
    tq.insert_all(tasks)
print("Done!")
