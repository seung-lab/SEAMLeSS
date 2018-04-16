import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample
import numpy as np
        
class Dense(nn.Module):
    def __init__(self, k=7, layers=4, fm=32, outfm=None, f=nn.ReLU(inplace=True), skip=True):
        super(Dense, self).__init__()
        p = int((k-1)/2)
        self.f = f
        self.skip = skip
        self.mlist = nn.ModuleList([nn.Conv2d(fm * (i+1), fm if outfm is None or i < layers - 1 else outfm, k, padding=p) for i in range(layers)])

    def forward(self, x):
        outputs = [x]
        for m in self.mlist:
            if m != self.mlist[-1] or not self.skip:
                outputs.append(self.f(m(torch.cat(outputs, 1))))
            else:
                outputs.append(m(torch.cat(outputs, 1)))
        return outputs[-1]

class PG(nn.Module):
    def __init__(self, k=7, layers=4, fm=32, f=nn.ReLU(inplace=True)):
        super(PG, self).__init__()
        print('building PG with kernel', k)
        p = int((k-1)/2)
        self.f = f
        self.encode = nn.Conv2d(2, 32, k, padding=p, groups=2)
        self.dense1 = Dense(fm=32, outfm=32, skip=False)
        self.dense2 = Dense(fm=32, outfm=16, skip=False)
        self.dense3 = Dense(fm=16, skip=False)
        self.decode = nn.Conv2d(16, 2, k, padding=p)

    def forward(self, x):
        embedding = self.f(self.encode(x))
        out1 = self.dense1(embedding)
        out2 = self.dense2(out1)
        out3 = self.dense3(out2)
        out = self.decode(out3).permute(0,2,3,1)
        return out / 10

class G(nn.Module):
    def __init__(self, k=7, f=nn.ReLU()):
        super(G, self).__init__()
        print('building G with kernel', k)
        p = int((k-1)/2)
        self.conv1 = nn.Conv2d(2, 32, k, padding=p, groups=2)
        self.conv2 = nn.Conv2d(32, 64, k, padding=p)
        self.conv3 = nn.Conv2d(64, 32, k, padding=p)
        self.conv4 = nn.Conv2d(32, 16, k, padding=p)
        self.conv5 = nn.Conv2d(16, 2, k, padding=p)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f,
                                 self.conv3, f,
                                 self.conv4, f,
                                 self.conv5)

    def forward(self, x):
        return self.seq(x).permute(0,2,3,1)

class Pyramid(nn.Module):
    def get_identity_grid(self, dim):
        if dim not in self.identities:
            gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
            I = np.stack(np.meshgrid(gx, gy))
            I = np.expand_dims(I, 0)
            I = torch.FloatTensor(I)
            I = torch.autograd.Variable(I, requires_grad=False)
            I = I.permute(0,2,3,1)
            self.identities[dim] = I.cuda()
        return self.identities[dim]

    def __init__(self, size, dim, skip, k, dilate=False, amp=False, unet=False):
        super(Pyramid, self).__init__()
        rdim = dim / (2 ** (size))
        print('------- Constructing PyramidNet with size', size, '(' + str(size-1) + ' downsamples)')
        self.identities = {}
        self.skip = skip
        self.size = size
        self.mlist = nn.ModuleList([PG(k=k) for level in range(size)])
        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)
        self.down = nn.AvgPool2d(2, 2)
        self.I = self.get_identity_grid(rdim)
        self.Zero = self.I - self.I

    def forward(self, stack, idx=0):
        if idx < self.size:
            field_so_far, residuals_so_far = self.forward(self.down(stack), idx + 1) # (B,dim,dim,2)
            field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)
        else:
            return self.I, [ self.I ]

        if idx < self.skip:
            residuals_so_far.insert(0, self.f_up(2 ** (self.size - idx))(self.Zero.permute(0,3,1,2)).permute(0,2,3,1)) # placeholder
            return field_so_far, residuals_so_far
        else:
            resampled_source = grid_sample(stack[:,0:1], field_so_far)
            new_input_stack = torch.cat((resampled_source, stack[:,1:2]),1)
            residual = self.mlist[idx](new_input_stack)
            residuals_so_far.insert(0, residual)
            return residual + field_so_far, residuals_so_far

class PyramidTransformer(nn.Module):
    def __init__(self, size=4, dim=192, skip=0, k=7, dilate=False, amp=False, unet=False):
        super(PyramidTransformer, self).__init__()
        self.pyramid = Pyramid(size, dim, skip, k, dilate, amp, unet)

    def open_layer(self):
        if self.pyramid.skip > 0:
            self.pyramid.skip -= 1
            print('Pyramid now using', self.pyramid.size - self.pyramid.skip, 'layers.')

    def select_module(self, idx):
        for g in self.pyramid.mlist:
            g.requires_grad = False
        self.pyramid.mlist[idx].requires_grad = True

    def select_all(self):
        for g in self.pyramid.mlist:
            g.requires_grad = True

    def forward(self, x, idx=0):
        field, residuals = self.pyramid(x, idx)
        #factor = 8
        #I =  self.pyramid.get_identity_grid(field.size()[2])
        #irfield = field - I
        #irfield_smooth = nn.AvgPool2d(2**factor+1, stride=1, padding=2**(factor-1), count_include_pad=False)(irfield.permute(0,3,1,2)).permute(0,2,3,1)
        #field = irfield_smooth + I
        return grid_sample(x[:,0:1,:,:], field), field, residuals

    ################################################################
    # Begin Sergiy API
    ################################################################

    @staticmethod
    def load(archive_path=None, height=6, dim=1024, skips=3, k=7, cuda=True, dilate=False, amp=False, unet=False):
        """
        Builds and load a model with the specified architecture from
        an archive.

        Params:
            height: the number of layers in the pyramid (including
                    bottom layer (number of downsamples = height - 1)
            dim:    the size of the full resolution images used as input
            skips:  the number of residual fields (from the bottom of the
                    pyramid) to skip
            cuda:   whether or not to move the model to the GPU
        """
        assert archive_path is not None, "Must provide an archive"

        model = PyramidTransformer(size=height, dim=dim, skip=skips, dilate=dilate, amp=amp, unet=unet)
        if cuda:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        model.train(False)

        print('Loading model state from', archive_path + '...')
        model.load_state_dict(torch.load(archive_path))

        return model

    def apply(self, source, target, skip=0):
        """
        Applies the model to an input. Inputs (source and target) are
        expected to be of shape (dim / (2 ** skip), dim / (2 ** skip)),
        where dim is the argument that was passed to the constructor.

        Params:
            source: the source image (to be transformed)
            target: the target image (image to align the source to)
            skip:   resolution at which alignment should occur.
        """
        return self.forward(torch.stack((source,target)).unsqueeze(0), idx=skip)
