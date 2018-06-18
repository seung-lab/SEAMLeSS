import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from helpers import reduce_seq, gif

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, f=F.elu, same=True, k=3):
        super(Conv, self).__init__()
        self.f = f
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, padding=(k-1)/2 if same else 0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=(k-1)/2 if same else 0)
        
    def forward(self, x):
        out = self.f(self.conv1(x))
        out = self.f(self.conv2(out))
        return out

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, f=F.elu, same=True):
        super(Down, self).__init__()
        self.f = f
        self.downsample = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1 if same else 0)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1 if same else 0)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1 if same else 0)

    def forward(self, x):
        out = self.f(self.downsample(x))
        out = self.f(self.conv1(out))
        out = self.f(self.conv2(out))
        return out

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, f=F.elu, same=True, cat=True):
        super(Up, self).__init__()
        self.f = f
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2, padding=1 if same else 0, output_padding=1)
        self.conv1 = nn.Conv2d(in_ch + out_ch if cat else in_ch, out_ch, 3, padding=1 if same else 0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1 if same else 0)
        
    def forward(self, x, y=None):
        out = self.f(self.upsample(x))
        if y is not None:
            out = reduce_seq((out, y), torch.cat)
        out = self.f(self.conv1(out))
        out = self.f(self.conv2(out))
        return out
        
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, depth=4, fm=64, same=True):
        super(UNet, self).__init__()

        self.skip = 0
        self.dropout = nn.Dropout2d(p=.15)
        
        self.downModules = nn.ModuleList()
        self.upModules = nn.ModuleList()
        for i in xrange(depth):
            self.downModules.append(Down(fm * (2 ** i), fm * (2 ** (i+1)), same=same))
            self.upModules.append(Up(fm * (2 ** (depth - i)), fm * (2 ** (depth - i - 1)), same=same, cat=i>=self.skip))

        self.embed_in = Conv(in_ch, fm, same=same, k=7)
        self.embed_out = Conv(fm, out_ch, same=same)
        
    def forward(self, x):
        out = self.embed_in(x.squeeze(1))
        #out = self.dropout(out)
        downPass = [out]
        upPass = []
        for idx, down in enumerate(self.downModules):
            out = down(out)
            downPass.append(out)
        for idx, up in enumerate(self.upModules):
            out = up(out, downPass[-(idx+2)]) if idx >= self.skip else up(out)
            upPass.append(out)
        out = self.embed_out(out).unsqueeze(1)
        return out
        
def save_act(self, a, name):
    return
    npy = a.data.cpu().numpy()
    for i in range(a.size()[1]):
        gif('out/' + name + '_ch' + str(i), npy[0,i,:,:,:])
    
