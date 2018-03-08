from aligner import Aligner, BoundingBox

a = Aligner('model/1_6_3.pt', (1024, 1024), 256, 6, 4, 'gs://neuroglancer/pinky40_v11/image', 'gs://neuroglancer/nflow_tests/p40')
t = 1024
s = 1024
bbox = BoundingBox(t, t + s, t, t + s)
a.align_ng_stack(10, 11, 0, bbox)
