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

class PreEnc(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, outfm=12):
        super(PreEnc, self).__init__()
        self.f = nn.LeakyReLU(inplace=True)
        self.c1 = nn.Conv2d(1, outfm // 2, 7, padding=3)
        self.c2 = nn.Conv2d(outfm // 2, outfm // 2, 7, padding=3)
        self.c3 = nn.Conv2d(outfm // 2, outfm, 7, padding=3)
        self.c4 = nn.Conv2d(outfm, outfm // 2, 7, padding=3)
        self.c5 = nn.Conv2d(outfm // 2, 1, 7, padding=3)
        self.initc(self.c1)
        self.initc(self.c2)
        self.initc(self.c3)
        self.initc(self.c4)
        self.initc(self.c5)
        self.pelist = nn.ModuleList([self.c1, self.c2, self.c3, self.c4, self.c5])

    def forward(self, x, vis=None):
        outputs = []
        for x_ch in range(x.size(1)):
            out = x[:,x_ch:x_ch+1]
            for idx, m in enumerate(self.pelist):
                out = m(out)
                if idx < len(self.pelist) - 1:
                    out = self.f(out)

            outputs.append(out)
        return torch.cat(outputs, 1)

class EPyramid(nn.Module):
    def get_identity_grid(self, dim, device):
        # print('get_identity_grid {0}'.format(device))
        # if dim not in self.identities:
        gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
        I = np.stack(np.meshgrid(gx, gy))
        I = np.expand_dims(I, 0)
        I = torch.tensor(I, dtype=torch.float32, device=device)
        # I = torch.autograd.Variable(I, requires_grad=False)
        I.requires_grad_(False)
        I = I.permute(0,2,3,1)
        # print('I in get_identity device: {0}'.format(I.device))
        # self.identities[dim] = I #.cuda(non_blocking=True)
        # return self.identities[dim]
        return I

    def __init__(self, size, dim, skip, topskips, k, num_targets=1, train_size=1280):
        super(EPyramid, self).__init__()
        rdim = dim // (2 ** (size - 1 - topskips))
        print('Constructing EPyramid with size {} ({} downsamples, input size {})...'.format(size, size-1, dim))
        fm_0 = 12
        fm_coef = 6
        self.identities = {}
        self.skip = skip
        self.topskips = topskips
        self.size = size
        self.dim = dim
        enc_outfms = [fm_0 + fm_coef * idx for idx in range(size)]
        enc_infms = [1] + enc_outfms[:-1]
        self.mlist = nn.ModuleList([G(k=k, infm=enc_outfms[level]*2) for level in range(size)])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = nn.MaxPool2d(2)
        self.enclist = nn.ModuleList([Enc(infm=infm, outfm=outfm) for infm, outfm in zip(enc_infms, enc_outfms)])
        # self.I = self.get_identity_grid(rdim)
        self.rdim = rdim
        self.TRAIN_SIZE = train_size
        self.pe = PreEnc(fm_0)

    def forward(self, stack, target_level, vis=None, use_preencoder=False):
        # print('forward device {0}'.format(stack.device))
        if vis is not None:
            gif(vis + 'input', gif_prep(stack))

        if use_preencoder:
            # run the preencoder
            residual = self.pe(stack)
            stack = stack + residual
            if vis is not None:
                # visualize the preencoder output
                zm = (stack == 0).data
                print('residual me,mi,ma {},{},{}'.format(torch.mean(residual[~zm]).data[0], torch.min(residual[~zm]).data[0], torch.max(residual[~zm]).data[0]))
                gif(vis + 'pre_enc_residual', gif_prep(residual))
                gif(vis + 'pre_enc_output', gif_prep(stack))
            if use_preencoder == "only":
                # only run the preencoder and return the results
                return stack

        encodings = [self.enclist[0](stack)]
        for idx in range(1, self.size-self.topskips):
            encodings.append(self.enclist[idx](self.down(encodings[-1]), vis=vis))

        # print('stack device: {0}'.format(stack.device))
        I = self.get_identity_grid(self.rdim, stack.device)
        residuals = [I]
        field_so_far = I * 0.0 # zero field
        # print('I device: {0}'.format(I.device))
        for i in range(self.size - 1 - self.topskips, target_level - 1, -1):
            if i >= self.skip:
                curr_dim = self.dim // (2 ** i)
                inputs_i = encodings[i]
                I = self.get_identity_grid(curr_dim, stack.device)
                resampled_source = grid_sample(inputs_i[:,0:inputs_i.size(1)//2],
                                             field_so_far + I, mode='bilinear')
                new_input_i = torch.cat((resampled_source, inputs_i[:,inputs_i.size(1)//2:]), 1)
                factor = ((self.TRAIN_SIZE / (2. ** i)) / (new_input_i.size()[-1] - 1))
                rfield = self.mlist[i](new_input_i) * factor
                residuals.append(rfield)
                # Resample residual at field_so_far using rfield correspondence.
                # Add result to rfield residual to produce the new
                # field_so_far residual.
                resampled_field_so_far = grid_sample(
                    field_so_far.permute(0,3,1,2),
                    rfield + self.get_identity_grid(self.dim // (2 ** i), stack.device),
                    mode='bilinear', padding_mode='border').permute(0,2,3,1)
                field_so_far = rfield + resampled_field_so_far
            if i != target_level:
                up_field = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)
                # account for shifting locations of -1 and +1 in upsampled field
                up_field *= (field_so_far.shape[2]-1.0)/field_so_far.shape[2]
                up_field /= (up_field.shape[2]-1.0)/up_field.shape[2]
                field_so_far = up_field
        curr_dim = self.dim // (2 ** target_level)
        field_so_far += self.get_identity_grid(curr_dim, stack.device)
        return field_so_far, residuals

class PyramidTransformer(nn.Module):
    def __init__(self, size=4, dim=192, skip=0, topskips=0, k=7, student=False, num_targets=1):
        super(PyramidTransformer, self).__init__()
        if not student:
            self.pyramid = EPyramid(size, dim, skip, topskips, k, num_targets)
        else:
            assert False # TODO: add student network

    @staticmethod
    def student(height, dim, skips, topskips, k):
        return PyramidTransformer(height, dim, skips, topskips, k, student=True).cuda()
    
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

    def forward(self, x, idx=0, vis=None, use_preencoder=False):
        if use_preencoder == "only":
            # only run the preencoder and return the results
            return self.pyramid(x, idx, vis, use_preencoder=use_preencoder)
        field, residuals = self.pyramid(x, idx, vis, use_preencoder=use_preencoder)
        return grid_sample(x[:,0:1,:,:], field, mode='bilinear'), field, residuals

    ################################################################
    # Begin Sergiy API
    ################################################################

    @staticmethod
    def load(archive_path=None, height=5, dim=1024, skips=0, topskips=0, k=7, cuda=True, num_targets=1):
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

        model = PyramidTransformer(size=height, dim=dim, k=k, skip=skips, topskips=topskips, num_targets=num_targets)
        if cuda:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        model.train(False)

        print('Loading model state from {}...'.format(archive_path))
        state_dict = torch.load(archive_path)
        copy_state_to_model(state_dict, model)
        print('Successfully loaded model state.')
        return model

    def apply(self, source, target, skip=0, vis=None, use_preencoder=False):
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
        return self(torch.cat((source,target), 0).unsqueeze(0), idx=skip, vis=vis, use_preencoder=use_preencoder)
