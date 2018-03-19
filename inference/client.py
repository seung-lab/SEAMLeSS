from aligner import Aligner, BoundingBox
<<<<<<< HEAD
from copy import deepcopy
from time import time
s = 64*(2**7)
t = 64*(2**7)
mip = 7
source_z = 11
target_z = 10
v_off = (10240, 4096, 0)

#patch_bbox = BoundingBox(v_off[0]+t, v_off[0]+t+s, v_off[1]+t, v_off[1]+t+s, mip=0, max_mip=9)
#patch_bbox = BoundingBox(v_off[0], v_off[0]+57344, v_off[1], v_off[1]+40960, mip=0, max_mip=9)
patch_bbox = BoundingBox(v_off[0]+t, v_off[0]+t+s, v_off[1]+t, v_off[1]+t+s, mip=0, max_mip=9)
crop = 64
influence_bbox = deepcopy(patch_bbox)
influence_bbox.uncrop(crop, mip=mip)

a = Aligner('model/2_5_2.pt', (64, 64), 1536, 32, 9, 9,
      'gs://neuroglancer/pinky40_alignment/prealigned',
      'gs://neuroglancer/nflow_tests/small_spoof_1024_0', move_anchor=True)

a.align_ng_stack(116, 118, patch_bbox)
for z in range(116, 118):
  a.set_processing_chunk_size((64, 64))
  start = time()
  #a.align_ng_stack(z, z + 1, patch_bbox)
  end = time()
  print ("Aligmnent: {} sec".format(end-start))
  c = 2048
  m = 2

  start = time()
  for x in range(3, 8):
    s = max(int(2048/(2**x)), 64)
    a.set_processing_chunk_size((s, s))
    a.render(z + 1, patch_bbox, mip=m+x)
  end = time()
  print ("Render: {} sec".format(end-start))
#a.compute_section_pair_residuals(source_z, target_z, patch_bbox)
#a.save_aggregate_flow(source_z, patch_bbox, mip=6)
#a.save_aggregate_flow(source_z, patch_bbox, mip=5)
#a.save_aggregate_flow(source_z, patch_bbox, mip=4)
#a.render(source_z, patch_bbox, mip=9)
#a.render(source_z, patch_bbox, mip=8)
#a.render(source_z, patch_bbox, mip=2)
#a.render(source_z, patch_bbox, mip=7)
=======

v_off = (10240, 4096, 0)
x_size = 57344
y_size = 40960

model_path = 'model/2_5_2.pt'
max_displacement = 2048
net_crop  = 32
mip_range = (7, 9)
render_mip = 4
high_mip_chunk = (64, 64)
a = Aligner('model/2_5_2.pt', max_displacement, net_crop, mip_range, render_mip, high_mip_chunk,
            'gs://neuroglancer/pinky40_alignment/prealigned',
            'gs://neuroglancer/nflow_tests/test')

bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=9)
stack_start = 30
stack_size  = 2
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)
stack_start += stack_size
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=False)
>>>>>>> e6f086e578aa4dd457c9ce5248eb82a2022307f2
