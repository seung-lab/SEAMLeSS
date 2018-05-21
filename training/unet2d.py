import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from helpers import reduce_seq, gif

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, f=F.relu, k=3, same=True, f_out=None, groups=1):
        super(Conv, self).__init__()
        self.f = f
        self.f_out = f_out
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, padding=(k-1)/2 if same else 0, groups=groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=(k-1)/2 if same else 0)
        
    def forward(self, x):
        out = self.f(self.conv1(x))
        out = self.f(self.conv2(out)) if self.f_out is None else self.f_out(self.conv2(out))
        return out

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, f=F.elu, k=3, same=True):
        super(Down, self).__init__()
        self.f = f
        self.downsample = nn.Conv2d(in_ch, in_ch, k, stride=2, padding=(k-1)/2 if same else 0)
        self.conv1 = nn.Conv2d(in_ch, in_ch, k, padding=(k-1)/2 if same else 0)
        self.conv2 = nn.Conv2d(in_ch, out_ch, k, padding=(k-1)/2 if same else 0)

    def forward(self, x):
        out = self.f(self.downsample(x))
        out = self.f(self.conv1(out))
        out = self.f(self.conv2(out))
        return out

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, f=F.relu, k=3, same=True, cat=True):
        super(Up, self).__init__()
        self.f = f
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, k, stride=2, padding=(k-1)/2 if same else 0, output_padding=1)
        self.conv1 = nn.Conv2d(in_ch + out_ch if cat else in_ch, out_ch, k, padding=(k-1)/2 if same else 0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=(k-1)/2 if same else 0)
        
    def forward(self, x, y=None):
        out = self.f(self.upsample(x))
        if y is not None:
            out = reduce_seq((out, y), torch.cat)
        out = self.f(self.conv1(out))
        out = self.f(self.conv2(out))
        return out
        
class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, depth=2, fm=16, k=3, same=True):
        super(UNet, self).__init__()

        print('--------- Building UNet with depth ' + str(depth) + ' and kernel size ' + str(k) + ' ---------')
        
        self.downModules = nn.ModuleList()
        self.upModules = nn.ModuleList()
        for i in xrange(depth):
            self.downModules.append(Down(fm, fm, k=k, same=same))
            self.upModules.append(Up(fm, fm, k=k, same=same))

        self.embed_in = Conv(in_ch, fm, same=same, groups=2)
        self.embed_out = Conv(fm, out_ch, same=same, f_out=lambda x: x)
        
    def forward(self, x):
        out = self.embed_in(x)
        downPass = [out]
        upPass = []
        for idx, down in enumerate(self.downModules):
            out = down(out)
            downPass.append(out)
        for idx, up in enumerate(self.upModules):
            out = up(out, downPass[-(idx+2)])
            upPass.append(out)
        out = self.embed_out(out)
        return out.permute(0,2,3,1)
