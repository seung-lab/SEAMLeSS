from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

from neuroflow.util import get_identity
import numpy as np
from conv import G

class Xmas(nn.Module):
    def __init__(self, levels=1, skip_levels=0, shape=[5,8,256,256]):
        super(Xmas, self).__init__()
        self.G_level = nn.ModuleList()
        self.levels = levels
        for i in range(levels):
            self.G_level.append(G)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ident = get_identity(batch_size=shape[1]-1, width=shape[3])
        self.ident = Variable(torch.from_numpy(self.ident).cuda(),
                                                requires_grad=False).permute(0,2,3,1)
        self.shape = shape
        self.shape[1] -= 1
        self.r_shape = [shape[0], 2, shape[1], shape[2], shape[3]]

    def up_crop(self, x): # [2,7,256,256]
        return self.upsample(x[:,:,64:64+128,64:64+128])

    def render(self, x, R): # [8,256,256], [2,7,256,256]
        R_p = R.permute(1,2,3,0) + self.ident
        y = F.grid_sample(x[1:,:,:].unsqueeze(1), R_p)
        return torch.cat([y.squeeze(1), x[-1].unsqueeze(0)], dim=0) #[8,256,256]

    def compute_res(self, x, level=0): # [8,256,256]
        # [7,2,256,256] Batchify
        batch = Variable(torch.zeros(x.shape[0]-1, 2, x.shape[-1], x.shape[-1]).cuda())
        for j in range(x.shape[0]-1):
            batch[j] = torch.stack([x[j], x[j+1]], 0)
        r = self.G_level[level](batch)
        return r

    def forward(self, xs): #[5,8,256,256,2]
        Rs = Variable(torch.zeros(self.r_shape).cuda())
        rs = Variable(torch.zeros(self.r_shape).cuda())
        ys = Variable(torch.zeros(self.shape).cuda())

        for i in range(self.levels):
            R = self.up_crop(Rs[i-1])
            x = self.render(xs[i,:,:,:], R) #[8,256,256]
            rs[i,:,:,:] = self.compute_res(x, i) #[7,256,256,2]
            Rs[i] = rs[i,:,:,:] + R
            ys[i] = self.render(xs[i,:,:,:], Rs[i])[:-1] #[7,256,256]
        return ys, Rs, rs #[7,256,256], [7,256,256,2], [7,256,256,2]

#Simple test
if __name__ == "__main__":
    flow = Xmas().cuda()
    s = np.ones((5,8,256,256), dtype=np.float32)
    x = torch.from_numpy(s).cuda()
    xs = torch.autograd.Variable(x, requires_grad=False)
    flow = flow(xs)
    print(len(flow)) #expected output (7,256,256,2)
