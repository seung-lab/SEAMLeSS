from aligner import Aligner, BoundingBox


model_path = 'model_repository/9_4_2.pt'
max_displacement = 2048
net_crop  = 64
mip_range = (6, 8)
render_mip = 2
high_mip_chunk = (256, 256)
a = Aligner(model_path, max_displacement, net_crop, mip_range, render_mip, high_mip_chunk, 'gs://neuroglancer/pinky40_alignment/prealigned_rechunked', 'gs://neuroglancer/nflow_tests/mip4fixed')

v_off = (10240, 4096, 0)
x_size = 57344
y_size = 40960
bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=8)

stack_start = 20
stack_size  = 30
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
