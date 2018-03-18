from aligner import Aligner, BoundingBox
from copy import deepcopy

s = 512
t = 5120
mip = 7
source_z = 11
target_z = 10
v_off = (10240, 4096, 0)

patch_bbox = BoundingBox(v_off[0]+t, v_off[0]+t+s, v_off[1]+t, v_off[1]+t+s, mip=0, max_mip=9)
patch_bbox = BoundingBox(v_off[0], v_off[0]+57344, v_off[1], v_off[1]+40960, mip=0, max_mip=9)
crop = 64
influence_bbox = deepcopy(patch_bbox)
influence_bbox.uncrop(crop, mip=mip)

a = Aligner('model/2_5_2.pt', (64, 64), 1024, 32, 6, 6, 'gs://neuroglancer/pinky40_alignment/prealigned', 'gs://neuroglancer/nflow_tests/p40_pre', move_anchor=True)

a.align_ng_stack(10, 10, patch_bbox)
#a.compute_section_pair_residuals(source_z, target_z, patch_bbox)
#a.save_aggregate_flow(source_z, patch_bbox, mip=6)
#a.save_aggregate_flow(source_z, patch_bbox, mip=5)
#a.save_aggregate_flow(source_z, patch_bbox, mip=4)
#for s in range(12, 21):
#    a.render(s, patch_bbox, mip=2)
#a.render(source_z, patch_bbox, mip=8)
#a.render(source_z, patch_bbox, mip=2)
#a.render(source_z, patch_bbox, mip=7)
