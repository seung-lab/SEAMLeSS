from aligner import Aligner, BoundingBox

a = Aligner('model/1_6_3.pt', (256, 256), 'gs://neuroglancer/pinky40_v11/image', 'gs://neuroglancer/nflow_tests/p40')
t = 1024
bbox = BoundingBox(t, t+ 256, t, t + 256)
a.align_ng_stack(0, 1, 4, bbox)
