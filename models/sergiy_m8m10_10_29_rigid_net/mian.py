from residuals import combine_residuals, upsample, downsample, shift_by_int
import torch
import sys

size = int(sys.argv[1])
x = torch.zeros([1, size, size, 2], dtype=torch.float32)
y = torch.zeros([1, size, size, 2], dtype=torch.float32)

x[...] = 1
y[...] = 1

z = combine_residuals(x, y, is_pix_res=True, rollback=3)
print (z.mean())

x_ = x.clone()
for i in range(5):
    x_ = upsample(x_, is_res=True)
for i in range(5):
    x_ = downsample(x_, is_res=True)
u = shift_by_int(x, 0, 1, is_res=True)
import pdb; pdb.set_trace()
