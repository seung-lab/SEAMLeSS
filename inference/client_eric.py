import sys
from aligner import Aligner, BoundingBox

model_name = sys.argv[1]
out_name = sys.argv[2]
model_path = 'model_repository/' + model_name + '.pt'
max_displacement = 2048
net_crop  = 256
mip_range = (3, 3)
render_mip = 3
high_mip_chunk = (1024, 1024)

a = Aligner(model_path, max_displacement, net_crop, mip_range, high_mip_chunk, 'gs://neuroglancer/pinky40_alignment/prealigned_rechunked', 'gs://neuroglancer/nflow_tests/' + model_name+'_'+out_name)

v_off = (10240*2, 4096*2, 0)
x_size = 57344/8
y_size = 40960/4
bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=9)

stack_start = 1
stack_size  = 100
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
