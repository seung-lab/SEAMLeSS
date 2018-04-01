from aligner import Aligner, BoundingBox

#v_off = (10240, 4096, 0)
v_off = (51200, 51200, 509)
x_size = 10*1024
y_size = 10*1024

model_path = 'model_repository/basil_7_1.pt'
max_displacement = 2048
net_crop  = 32
mip_range = (4, 8)
render_mip = 4
high_mip_chunk = (64, 64)

a = Aligner(model_path, max_displacement,
            net_crop, mip_range, render_mip, high_mip_chunk,
            'gs://neuroglancer/basil_v0/raw_image',
            'gs://neuroglancer/nflow_tests/davit_test',
            is_Xmas = True)

bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=9)
stack_start = 1
stack_size  = 2
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
