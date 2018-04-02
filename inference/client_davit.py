from aligner import Aligner, BoundingBox

#v_off = (51200, 51200, 0)
v_off = (10240, 4096, 0)
x_size = 57344
y_size = 40960

model_path = 'model_repository/basil2_7_1.pt'
max_displacement = 2048
net_crop  = 32
mip_range = (5, 8)
render_mip = 5
high_mip_chunk = (64, 64)

a = Aligner(model_path, max_displacement,
            net_crop, mip_range, render_mip, high_mip_chunk,
            'gs://neuroglancer/pinky40_alignment/prealigned_rechunked',
            'gs://neuroglancer/nflow_tests/davit_test_6',
            is_Xmas = True)

bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=9)
stack_start = 766
stack_size  = 1
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
