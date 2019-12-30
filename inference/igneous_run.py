import sys
from taskqueue import TaskQueue
import igneous.task_creation as tc
from cloudvolume import Bbox

with TaskQueue('sergiy-fly-1') as tq:
    tasks = tc.create_downsampling_tasks(
            'precomputed://gs://fafb_v15_montages/sergiy_playground/main_region',
        chunk_size=[1024, 1024, 1],
        fill_missing=True,
        bounds=Bbox((0, 0, 3100), (231424*8*1024, 114688+8*1024, 3400)),
        mip=0,
        num_mips=6,
        preserve_chunk_size=True,
        delete_black_uploads=True
    )
    tq.insert_all(tasks)
print("Done!")
