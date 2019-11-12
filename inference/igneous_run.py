import sys
from taskqueue import TaskQueue
import igneous.task_creation as tc
from cloudvolume import Bbox

with TaskQueue('deepalign-igneous-1') as tq:

  tasks = tc.create_downsampling_tasks(
    'precomputed://gs://seunglab_minnie_phase3/alignment/fine_inference_x2_optim300_lr3em2_sm25e1_maskthresh07_mse3_vv5_long/warped_em_fromcoarse',
    chunk_size=[1024, 1024, 1],
    fill_missing=True,
    bounds=Bbox((0, 0, 17405), (524288, 393216, 17600)),
    mip=2,
    num_mips=6,
    preserve_chunk_size=True,
    delete_black_uploads=True
)
tq.insert_all(tasks)
print("Done!")
