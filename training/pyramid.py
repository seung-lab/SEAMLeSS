import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample
import numpy as np
from helpers import save_chunk, gif, copy_state_to_model
import random

class G(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, k=7, f=nn.LeakyReLU(inplace=True), infm=2):
        super(G, self).__init__()
        p = (k-1)//2
        self.conv1 = nn.Conv2d(infm, 32, k, padding=p)
        self.conv2 = nn.Conv2d(32, 64, k, padding=p)
        self.conv3 = nn.Conv2d(64, 32, k, padding=p)
        self.conv4 = nn.Conv2d(32, 16, k, padding=p)
        self.conv5 = nn.Conv2d(16, 2, k, padding=p)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f,
                                 self.conv3, f,
                                 self.conv4, f,
                                 self.conv5)
        self.initc(self.conv1)
        self.initc(self.conv2)
        self.initc(self.conv3)
        self.initc(self.conv4)
        self.initc(self.conv5)
        
    def forward(self, x):
        return self.seq(x).permute(0,2,3,1) / 10

class GS(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, k=7, f=nn.LeakyReLU(inplace=True), infm=2):
        super(GS, self).__init__()
        k = 3
        p = (k-1)//2
        self.conv1 = nn.Conv2d(infm, 16, 7, padding=3)
        self.conv2 = nn.Conv2d(16, 24, 5, padding=2)
        self.conv3 = nn.Conv2d(24, 8, k, padding=p)
        self.conv4 = nn.Conv2d(8, 8, k, padding=p)
        self.conv5 = nn.Conv2d(8, 8, k, padding=p)
        self.conv6 = nn.Conv2d(8, 8, k, padding=p)
        self.conv7 = nn.Conv2d(8, 2, k, padding=p)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f,
                                 self.conv3, f,
                                 self.conv4, f,
                                 self.conv5, f,
                                 self.conv6, f,
                                 self.conv7)
        self.initc(self.conv1)
        self.initc(self.conv2)
        self.initc(self.conv3)
        self.initc(self.conv4)
        self.initc(self.conv5)
        self.initc(self.conv6)
        self.initc(self.conv7)
        
    def forward(self, x):
        out = self.seq(x).permute(0,2,3,1) / 10
        return out
        
def gif_prep(s):
    if type(s) != np.ndarray:
        s = np.squeeze(s.data.cpu().numpy())
    for slice_idx in range(s.shape[0]):
        s[slice_idx] -= np.min(s[slice_idx])
        s[slice_idx] *= 255 / np.max(s[slice_idx])
    return s

class Enc(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)
        
    def __init__(self, infm, outfm):
        super(Enc, self).__init__()
        if not outfm:
            outfm = infm
        self.f = nn.LeakyReLU(inplace=True)
        self.c1 = nn.Conv2d(infm, outfm, 3, padding=1)
        self.c2 = nn.Conv2d(outfm, outfm, 3, padding=1)
        self.initc(self.c1)
        self.initc(self.c2)
        self.infm = infm
        self.outfm = outfm
        
    def forward(self, x, vis=None):
        ch = x.size(1)
        ngroups = ch // self.infm
        ingroup_size = ch//ngroups
        input_groups = [self.f(self.c1(x[:,idx*ingroup_size:(idx+1)*ingroup_size])) for idx in range(ngroups)]
        out1 = torch.cat(input_groups, 1)
        input_groups2 = [self.f(self.c2(out1[:,idx*self.outfm:(idx+1)*self.outfm])) for idx in range(ngroups)]
        out2 = torch.cat(input_groups2, 1)
        
        if vis is not None:
            visinput1, visinput2 = gif_prep(out1), gif_prep(out2)
            gif(vis + '_out1_' + str(self.infm), visinput1)    
            gif(vis + '_out2_' + str(self.infm), visinput2)
            
        return out2

class EncS(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)
        
    def __init__(self, infm, outfm):
        super(EncS, self).__init__()
        if not outfm:
            outfm = infm
        self.f = nn.LeakyReLU(inplace=True)
        self.c1 = nn.Conv2d(infm, outfm, 5, padding=2)
        self.c2 = nn.Conv2d(outfm, outfm, 5, padding=2)
        self.c3 = nn.Conv2d(outfm, outfm, 5, padding=2)
        self.initc(self.c1)
        self.initc(self.c2)
        self.infm = infm
        self.outfm = outfm
        
    def forward(self, x, vis=None):
        ch = x.size(1)
        ngroups = ch // self.infm
        ingroup_size = ch//ngroups
        input_groups = [self.f(self.c1(x[:,idx*ingroup_size:(idx+1)*ingroup_size])) for idx in range(ngroups)]
        out1 = torch.cat(input_groups, 1)
        input_groups2 = [self.f(self.c2(out1[:,idx*self.outfm:(idx+1)*self.outfm])) for idx in range(ngroups)]
        out2 = torch.cat(input_groups2, 1)
        input_groups3 = [self.f(self.c3(out2[:,idx*self.outfm:(idx+1)*self.outfm])) for idx in range(ngroups)]
        out3 = torch.cat(input_groups3, 1)
        
        if vis is not None:
            visinput1, visinput2, visinput3 = gif_prep(out1), gif_prep(out2), gif_prep(out3)
            gif(vis + '_out1_' + str(self.infm), visinput1)    
            gif(vis + '_out2_' + str(self.infm), visinput2)
            gif(vis + '_out3_' + str(self.infm), visinput3)
            
        return out3
    
class EPyramid(nn.Module):
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
    
    def __init__(self, size, dim, skip, topskips, k, dilate=False, amp=False, unet=False, num_targets=1, name=None, train_size=1280, target_weights=None):
        super(EPyramid, self).__init__()
        rdim = dim // (2 ** (size - 1 - topskips))
        print('------- Constructing EPyramid with size', size, '(' + str(size-1) + ' downsamples) ' + str(dim))
        if name:
            self.name = name
        fm_0 = 12
        fm_coef = 6
        self.identities = {}
        self.skip = skip
        self.topskips = topskips
        self.size = size
        enc_infms = [1] + [fm_0 + fm_coef * idx for idx in range(size-1)]
        enc_outfms = enc_infms[1:] + [fm_0 + fm_coef * (size-1)]
        num_slices = 1 + (num_targets if target_weights is None else 1)
        self.mlist = nn.ModuleList([G(k=k, infm=enc_outfms[level]*num_slices) for level in range(size)])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = nn.MaxPool2d(2)
        self.enclist = nn.ModuleList([Enc(infm=infm, outfm=outfm) for infm, outfm in zip(enc_infms, enc_outfms)])
        self.I = self.get_identity_grid(rdim)
        self.TRAIN_SIZE = train_size
        self.num_targets = num_targets
        self.target_weights = target_weights
        if target_weights is not None:
            assert len(target_weights) == num_targets
        
    def forward(self, stack, target_level, vis=None):
        if vis is not None:
            gif(vis + 'input', gif_prep(stack))
        
        encodings = [self.enclist[0](stack)]
        for idx in range(1, self.size-self.topskips):
            encodings.append(self.enclist[idx](self.down(encodings[-1]), vis=vis))

        if self.target_weights is not None:
            for idx, e in enumerate(encodings):
                chunk_length = e.size(1) // (self.num_targets + 1)
                chunks = [e[:,chunk_length * i:chunk_length * (i+1)] for i in range(self.num_targets+1)]
                encodings[idx] = torch.cat((chunks[0], sum([c * self.target_weights[i] for i, c in enumerate(chunks[1:])])), 1)

        residuals = [self.I]
        field_so_far = self.I
        for i in range(self.size - 1 - self.topskips, target_level - 1, -1):
            if i >= self.skip:
                inputs_i = encodings[i]
                resampled_source = grid_sample(inputs_i[:,0:inputs_i.size(1)//2], field_so_far, mode='bilinear')
                new_input_i = torch.cat((resampled_source, inputs_i[:,inputs_i.size(1)//2:]), 1)
                factor = ((self.TRAIN_SIZE / (2. ** i)) / new_input_i.size()[-1])
                rfield = self.mlist[i](new_input_i) * factor
                residuals.append(rfield)
                field_so_far = rfield + field_so_far
            if i != target_level:
                field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)
        return field_so_far, residuals

class SEPyramid(nn.Module):
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
    
    def __init__(self, size, dim, skip, topskips, k, dilate=False, amp=False, unet=False, num_targets=1, name=None, train_size=1280, target_weights=None):
        super(SEPyramid, self).__init__()
        rdim = dim // (2 ** (size - 1 - topskips))
        print('------- Constructing SEPyramid with size', size, '(' + str(size-1) + ' downsamples) ' + str(dim))
        if name:
            self.name = name
        fm_0 = 5
        fm_coef = 10
        self.identities = {}
        self.skip = skip
        self.topskips = topskips
        self.size = size
        enc_infms = [1] + [fm_0 + fm_coef * idx for idx in range(size-1)]
        enc_outfms = enc_infms[1:] + [fm_0 + fm_coef * (size-1)]
        num_slices = 1 + (num_targets if target_weights is None else 1)
        self.mlist = nn.ModuleList([GS(k=k, infm=enc_outfms[level]*num_slices) for level in range(size)])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = nn.MaxPool2d(2)
        self.enclist = nn.ModuleList([EncS(infm=infm, outfm=outfm) for infm, outfm in zip(enc_infms, enc_outfms)])
        self.I = self.get_identity_grid(rdim)
        self.TRAIN_SIZE = train_size
        self.num_targets = num_targets
        self.target_weights = target_weights
        if target_weights is not None:
            assert len(target_weights) == num_targets
        
    def forward(self, stack, target_level, vis=None):
        if vis is not None:
            gif(vis + 'input', gif_prep(stack))
        
        encodings = [self.enclist[0](stack)]
        for idx in range(1, self.size-self.topskips):
            encodings.append(self.enclist[idx](self.down(encodings[-1]), vis=vis))

        if self.target_weights is not None:
            for idx, e in enumerate(encodings):
                chunk_length = e.size(1) // (self.num_targets + 1)
                chunks = [e[:,chunk_length * i:chunk_length * (i+1)] for i in range(self.num_targets+1)]
                encodings[idx] = torch.cat((chunks[0], sum([c * self.target_weights[i] for i, c in enumerate(chunks[1:])])), 1)

        residuals = [self.I]
        field_so_far = self.I
        for i in range(self.size - 1 - self.topskips, target_level - 1, -1):
            if i >= self.skip:
                inputs_i = encodings[i]
                resampled_source = grid_sample(inputs_i[:,0:inputs_i.size(1)//2], field_so_far, mode='bilinear')
                new_input_i = torch.cat((resampled_source, inputs_i[:,inputs_i.size(1)//2:]), 1)
                factor = ((self.TRAIN_SIZE / (2. ** i)) / new_input_i.size()[-1])
                rfield = self.mlist[i](new_input_i) * factor
                residuals.append(rfield)
                field_so_far = rfield + field_so_far
            if i != target_level:
                field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)
        return field_so_far, residuals

    
class PyramidTransformer(nn.Module):
    def __init__(self, size=4, dim=192, skip=0, topskips=0, k=7, dilate=False, amp=False, unet=False, num_targets=1, name=None, target_weights=None, student=False):
        super(PyramidTransformer, self).__init__()
        if not student:
            self.pyramid = EPyramid(size, dim, skip, topskips, k, dilate, amp, unet, num_targets, name=name, target_weights=target_weights)
        else:
            self.pyramid = SEPyramid(size, dim, skip, topskips, k)

    def select_module(self, idx):
        for g in self.pyramid.mlist:
            g.requires_grad = False
        self.pyramid.mlist[idx].requires_grad = True

    def select_all(self):
        for g in self.pyramid.mlist:
            g.requires_grad = True

    def forward(self, x, idx=0, vis=None):
        field, residuals = self.pyramid(x, idx, vis)
        return grid_sample(x[:,0:1,:,:], field, mode='nearest'), field, residuals

    @staticmethod
    def student(height, dim, skips, topskips, k):
        return PyramidTransformer(height, dim, skips, topskips, k, student=True).cuda()
    
    ################################################################
    # Begin Sergiy API
    ################################################################

    @staticmethod
    def load(archive_path=None, height=5, dim=1024, skips=0, topskips=0, k=7, cuda=True, dilate=False, amp=False, unet=False, num_targets=1, name=None, target_weights=None):
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

        model = PyramidTransformer(size=height, dim=dim, k=k, skip=skips, topskips=topskips, dilate=dilate, amp=amp, unet=unet, num_targets=num_targets, name=name, target_weights=target_weights)
        if cuda:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        model.train(False)

        print('Loading model state from ' + archive_path + '...')
        state_dict = torch.load(archive_path)
        copy_state_to_model(state_dict, model) #model.load_state_dict(state_dict)
        print('Successfully loaded model state.')
        return model

    def apply(self, source, target, skip=0, vis=None):
        """
        Applies the model to an input. Inputs (source and target) are
        expected to be of shape (dim // (2 ** skip), dim // (2 ** skip)),
        where dim is the argument that was passed to the constructor.

        Params:
            source: the source image (to be transformed)
            target: the target image (image to align the source to)
            skip:   resolution at which alignment should occur.
        """
        source = source.unsqueeze(0)
        if len(target.size()) == 2:
            target = target.unsqueeze(0)
        return self.forward(torch.cat((source,target), 0).unsqueeze(0), idx=skip, vis=vis)
