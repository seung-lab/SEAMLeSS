import inspect_field as IF
from cloudvolume.lib import Vec, Bbox

cv_path = 'gs://neuroglancer/basil_v0/son_of_alignment/v3.04/optimizer_tests/refactor_invert_smoothness_v1/field'
f = IF.Field(cv_path, port=9997)

full_offset = Vec(102716, 107077, 526)
# full_offset = Vec(102320, 111106, 526)
full_size = Vec(2048, 2048, 1)
full_bbox = Bbox(full_offset, full_offset+full_size)

inspect_offset = Vec(103469, 107834, 526)
inspect_size = Vec(512, 512, 1)
inspect_bbox = Bbox(inspect_offset, inspect_offset+inspect_size)

I = f.inspect(inspect_bbox, full_bbox, mip=6)