import sys
from taskqueue import TaskQueue
import igneous.task_creation as tc
from cloudvolume import Bbox

with TaskQueue('deepalign-igneous-1') as tq:

  tasks = tc.create_downsampling_tasks(
    'precomputed://gs://seunglab_minnie_phase3/alignment/precoarse_vv5_tempdiv4_step128_maxdisp128/warped_result',
    chunk_size=[1024, 1024, 1],
    fill_missing=True,
    bounds=Bbox((0, 0, 17000), (524288, 393216, 18000)),
    mip=4,
    num_mips=4,
    preserve_chunk_size=True,
    delete_black_uploads=True
)
tq.insert_all(tasks)
print("Done!")
