from aligner import Aligner, BoundingBox

model_name = '11_3_0'
model_path = 'model_repository/' + model_name + '.pt'
max_displacement = 2048
net_crop  = 64
mip_range = (3, 7)
render_mip = 0
high_mip_chunk = (64, 64)
a = Aligner(model_path, max_displacement, net_crop, mip_range, render_mip, high_mip_chunk, 'gs://neuroglancer/pinky40_alignment/prealigned_rechunked', 'gs://neuroglancer/nflow_tests/' + model_name)

v_off = (10240, 4096, 0)
x_size = 57344
y_size = 40960
bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=7)

stack_start = 17
stack_size  = 30
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
