import sys
from aligner import Aligner, BoundingBox

model_name = sys.argv[1]
out_name = sys.argv[2]
params = int(sys.argv[3])
mip = int(sys.argv[4])
render_mip = int(sys.argv[5])
should_contrast = bool(int(sys.argv[6]))
model_path = 'model_repository/' + model_name + '.pt'
max_displacement = 2048
net_crop  = 384
mip_range = (mip,mip)
high_mip_chunk = (1024, 1024)

print('Model name:', model_name)
print('Tag:', out_name)
print('Param set:', params)
print('Mip:', mip)
print('Contrast:', should_contrast)

max_mip = 9
if params == 0:
    # basil test
    source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
    v_off = (102400, 102400, 179)
    x_size = 10240*4
    y_size = 10240*4
elif params == 1:
    # basil folds
    # gs://neuroglancer/basil_v0/father_of_alignment/v3
    source = 'gs://neuroglancer/nflow_tests/bprodsmooth_crack_pass/image'
    v_off = (10240*18, 10240*4, 179)
    x_size = 1024*16
    y_size = 1024*16
elif params == 2:
    # fly normal
    source = 'gs://neuroglancer/drosophila_v0/image_v14_single_slices'
    v_off = (10240*12, 10240 * 4, 2410)
    x_size = 10240 * 4
    y_size = 10240 * 4
    max_mip = 6
elif params == 3:
    # basil big
    source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
    v_off = (10240*4, 10240*2, 179)
    x_size = 10240*16
    y_size = 10240*16
print('Max mip:', max_mip-1)
print('NG link:', '\nprecomputed://' + 'gs://neuroglancer/seamless/' + model_name+'_'+out_name+'/image')

a = Aligner(model_path, max_displacement, net_crop, mip_range, high_mip_chunk, source, 'gs://neuroglancer/seamless/' + model_name+'_'+out_name, render_low_mip=render_mip, skip=0, topskip=0, should_contrast=should_contrast)

bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=max_mip)

stack_start = v_off[2]
stack_size  = 500
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
