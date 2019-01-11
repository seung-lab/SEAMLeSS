import torch
import torch.nn as nn
import copy
from utilities.helpers import (gridsample_residual, upsample, downsample,
                               load_model_from_dict, save_chunk, gif)
import numpy as np


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, height, dim=1536, feature_maps=None, *args, **kwargs):
        super().__init__()
        self.height = height
        self.encode = None
        self.align = PyramidTransformer(size=height, dim=dim, *args, **kwargs)

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, src, tgt, in_field=None, encodings=False, **kwargs):
        if self.encode is not None and encodings:
            src, tgt = self.encode(src, tgt)
        field = self.align.apply(src, tgt)
        return field

    def load(self, path):
        """
        Loads saved weights into the model
        """
        with path.open('rb') as f:
            weights = torch.load(f)
        load_model_from_dict(self.align, weights)
        # model_params = dict(self.align.named_parameters())
        # model_keys = sorted(model_params.keys())
        # print(model_keys)
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        with path.open('wb') as f:
            torch.save(self.align.state_dict(), f)

    def submodule(self, index):
        """
        Returns a submodule as indexed by `index`.

        Submodules with lower indecies are intended to be trained earlier,
        so this also decides the training order.

        `index` must be an int, a slice, or None.
        If `index` is a slice, the submodule contains the relevant levels.
        If `index` is None or greater than the height, the submodule
        returned contains the whole model.
        """
        if ((isinstance(index, int) and index >= self.height)
                or index.stop is None or index.stop >= self.height):
            for p in self.parameters():
                p.requires_grad = True
            self.train_all = lambda: self
            self.train_last = lambda: self
            self.pixel_size_ratio = 0
            return self
        else:
            raise NotImplementedError()


class Encoder(nn.Module):
    """
    Module that implements a two-convolution siamese encoder.
    These can be stacked to build an encoding pyramid.
    """

    def __init__(self, infm, outfm, k=3):
        super().__init__()
        p = (k-1)//2
        self.seq = nn.Sequential(
            nn.Conv2d(infm, outfm, k, padding=p),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outfm, outfm, k, padding=p),
            nn.LeakyReLU(inplace=True),
        )
        self.seq.apply(init_leaky_relu)

    def forward(self, src, tgt):
        return self.seq(src), self.seq(tgt)


class EncodingPyramid(nn.Module):
    """
    A stack of siamese encoders with one Encoder module at each mip level.
    It takes a pair of images and returns a list of encodings, one for
    each element of `feature_list`

    `feature_list` should be a list of integers, each of which specifies
    the number of feature maps at a particular mip level.
    For example,
        >>> EncodingPyramid([1, 2, 4, 8])
    creates a pyramid with four Encoder modules, with 1, 2, 4, and 8
    feature maps respectively.
    """

    def __init__(self, feature_list):
        super().__init__()
        self.feature_list = list(feature_list)
        self.list = nn.ModuleList([
            Encoder(infm, outfm)
            for infm, outfm
            in zip(self.feature_list[:-1], self.feature_list[1:])
        ])

    def forward(self, src, tgt):
        src_encodings = []
        tgt_encodings = []
        for module in self.list:
            src, tgt = module(src, tgt)
            src_encodings.append(src)
            tgt_encodings.append(tgt)
            src, tgt = downsample()(src), downsample()(tgt)
        return src_encodings, tgt_encodings


class Aligner(nn.Module):
    """
    Module that takes a pair of images as input and outputs a vector field
    that can be used to transform one to the other.

    While the output of the module has the standard shape for input to
    the PyTorch gridsampler, the units of the field is pixels in order
    to be agnostic to the size of the input images.
    """

    def __init__(self, channels=1, k=7):
        super().__init__()
        p = (k-1)//2
        self.channels = channels
        self.seq = nn.Sequential(
            nn.Conv2d(channels * 2, 16, k, padding=p),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, k, padding=p),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, k, padding=p),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, k, padding=p),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 2, k, padding=p),
        )
        self.seq.apply(init_leaky_relu)

    def forward(self, src, tgt):
        if src.shape[1] == 1:  # single images
            stack = torch.cat((src, tgt), dim=1)
            field = self.seq(stack).permute(0, 2, 3, 1)
            return field
        else:  # stack of encodings  # TODO: improve handling
            fields = []
            for pair in zip(src.split(self.channels, dim=1),
                            tgt.split(self.channels, dim=1)):
                stack = torch.cat(pair, dim=1)
                fields.append(self.seq(stack).permute(0, 2, 3, 1))
            return sum(fields) / len(fields)


class AligningPyramid(nn.Module):
    """
    A stack of Aligner modules with one Aligner at each mip level.
    It takes a pair of images and produces a vector field which maps from
    the first image, the source, to the second image, the target.

    If `src_input` and `tgt_input` are lists, then they are taken to be
    precomputed encodings or downsamples of the respective images.
    """

    def __init__(self, height):
        super().__init__()
        self.height = height
        self.list = nn.ModuleList([Aligner() for __ in range(height)])

    def forward(self, src_input, tgt_input, accum_field=None):
        for i in reversed(range(self.height)):
            if isinstance(src_input, list) and isinstance(tgt_input, list):
                src, tgt = src_input[i], tgt_input[i]
            else:
                src, tgt = downsample(i)(src_input), downsample(i)(tgt_input)
            if accum_field is not None:
                accum_field = (upsample()(accum_field.permute(0, 3, 1, 2))
                               .permute(0, 2, 3, 1))
                src = gridsample_residual(src, accum_field,
                                          padding_mode='border')
            factor = 2 / src.shape[-1]  # scale to [-1,1]
            res_field = self.list[i](src, tgt) * factor
            if accum_field is not None:
                resampled = gridsample_residual(
                    accum_field.permute(0, 3, 1, 2), res_field,
                    padding_mode='border').permute(0, 2, 3, 1)
                accum_field = res_field + resampled
            else:
                accum_field = res_field
        return accum_field


class _SubmoduleView(nn.Module):
    """
    Returns a view into a sequence of aligners of a model.
    This is useful for training and testing.

    This can be modified later to also include encodings.
    """

    def __init__(self, model, index):
        super().__init__()
        if isinstance(index, int):
            index = slice(index, index+1)
        self.levels = range(model.height)[index]
        self.aligners = model.align.list[index]

    def forward(self, src_input, tgt_input, accum_field=None):
        prev_level = None
        for i, aligner in zip(reversed(self.levels), reversed(self.aligners)):
            if isinstance(src_input, list) and isinstance(tgt_input, list):
                src, tgt = src_input[i], tgt_input[i]
            else:
                src, tgt = downsample(i)(src_input), downsample(i)(tgt_input)
            if prev_level is not None:
                accum_field = (upsample(prev_level - i)
                               (accum_field.permute(0, 3, 1, 2))
                               .permute(0, 2, 3, 1))
                src = gridsample_residual(src, accum_field,
                                          padding_mode='border')
            factor = 2 / src.shape[-1]  # scale to [-1,1]
            res_field = aligner(src, tgt) * factor
            if accum_field is not None:
                resampled = gridsample_residual(
                    accum_field.permute(0, 3, 1, 2), res_field,
                    padding_mode='border').permute(0, 2, 3, 1)
                accum_field = res_field + resampled
            else:
                accum_field = res_field
            prev_level = i
        accum_field = (upsample(prev_level)
                       (accum_field.permute(0, 3, 1, 2))
                       .permute(0, 2, 3, 1))
        return accum_field

    def train_all(self):
        """
        Train all the levels of the submodule
        """
        for p in self.parameters():
            p.requires_grad = True
        return self

    def train_last(self):
        """
        Train only the final level of the submodule and freeze
        all the other weights
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.aligners[-1].parameters():
            p.requires_grad = True
        return self

    def init_last(self):
        """
        Initialize the last level of the submodule by copying the trained
        weights of the previous level.
        If the submodule has only one level, this does nothing.
        """
        if len(self.aligners) > 1:
            state_dict = self.aligners[-2].state_dict()
            self.aligners[-1].load_state_dict(state_dict)
        return self

    @property
    def pixel_size_ratio(self):
        """
        The ratio of the pixel size of the submodule's highest level to
        the pixel size at its lowest level.
        By assumption, each level of the network has equal ability, so this
        is a measure of the power of the submodule to detect and correct
        large misalignments in its input scale.
        """
        return 2**(self.levels[-1] - self.levels[0])


def init_leaky_relu(m, a=None):
    """
    Initialize to account for the default negative slope of LeakyReLU.
    PyTorch's LeakyReLU by defualt has a slope of 0.01 for negative
    values, but the default initialization for Conv2d uses
    `kaiming_uniform_` with `a=math.sqrt(5)`. (ref: https://goo.gl/Bx3wdS)
    Instead, this initializes according to He, K. et al. (2015).
    (ref https://goo.gl/hH6qaM)

    If `a` is given it uses that as the negative slope. If it is None,
    the default for LeakyReLU is used.
    """
    if not isinstance(m, torch.nn.Conv2d):
        return
    if a is None:
        a = nn.modules.activation.LeakyReLU().negative_slope
    nn.init.kaiming_uniform_(m.weight, a=a)


# helper functions kept around temporarily... TODO: remove

def copy_aligner(self, id_from, id_to):
    """
    Copy the kernel weights from one aligner module to another
    """
    if min(id_from, id_to) < 0 or max(id_from, id_to) >= self.height:
        raise IndexError('Values {} --> {} out of bounds for size {}.'
                         .format(id_from, id_to, self.height))
    state_dict = self.align.list[id_from].state_dict()
    self.align.list[id_to].load_state_dict(state_dict)


def shift_aligners(self):
    """
    Shift the kernel weights up one aligner and make a copy of the lowest
    """
    for i in range(self.height-1, 1, -1):
        self.align.list[i] = self.align.list[i-1]
    self.align.list[1] = copy.deepcopy(self.align.list[0])


def copy_encoder(self, id_from, id_to):
    """
    Copy the kernel weights from one encoder module to another
    """
    if min(id_from, id_to) < 0 or max(id_from, id_to) >= self.height:
        raise IndexError('Values {} --> {} out of bounds for size {}.'
                         .format(id_from, id_to, self.height))
    state_dict = self.encode.list[id_from].state_dict()
    self.encode.list[id_to].load_state_dict(state_dict)


# Old pyramid code, wrapped by Model()

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
        return self.seq(x).permute(0, 2, 3, 1) / 10


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
        input_groups = [self.f(self.c1(x[
            :, idx*ingroup_size:(idx+1)*ingroup_size]))
            for idx in range(ngroups)]
        out1 = torch.cat(input_groups, 1)
        input_groups2 = [self.f(self.c2(out1[
            :, idx*self.outfm:(idx+1)*self.outfm])) for idx in range(ngroups)]
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
        self.pelist = nn.ModuleList(
            [self.c1, self.c2, self.c3, self.c4, self.c5])

    def forward(self, x, vis=None):
        outputs = []
        for x_ch in range(x.size(1)):
            out = x[:, x_ch:x_ch+1]
            for idx, m in enumerate(self.pelist):
                out = m(out)
                if idx < len(self.pelist) - 1:
                    out = self.f(out)

            outputs.append(out)
        return torch.cat(outputs, 1)


class EPyramid(nn.Module):
    def __init__(self, size, dim, skip, topskips, k, num_targets=1,
                 train_size=1280):
        super(EPyramid, self).__init__()
        print('Constructing EPyramid with size {} ({} downsamples, '
              'input size {})...'.format(size, size-1, dim))
        fm_0 = 12
        fm_coef = 6
        self.identities = {}
        self.skip = skip
        self.topskips = topskips
        self.size = size
        self.dim = dim
        self.rdim = dim // (2 ** (size - 1 - topskips))
        enc_outfms = [fm_0 + fm_coef * idx for idx in range(size)]
        enc_infms = [1] + enc_outfms[:-1]
        self.mlist = nn.ModuleList([G(k=k, infm=enc_outfms[level]*2)
                                    for level in range(size)])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = nn.MaxPool2d(2)
        self.enclist = nn.ModuleList([Enc(infm=infm, outfm=outfm)
                                      for infm, outfm
                                      in zip(enc_infms, enc_outfms)])
        self.TRAIN_SIZE = train_size
        self.pe = PreEnc(fm_0)

    def forward(self, stack, target_level, vis=None, use_preencoder=False):
        if vis is not None:
            gif(vis + 'input', gif_prep(stack))

        if use_preencoder:
            # run the preencoder
            residual = self.pe(stack)
            stack = stack + residual
            if vis is not None:
                # visualize the preencoder output
                zm = (stack == 0).data
                print('residual me,mi,ma {},{},{}'.format(
                    torch.mean(residual[~zm]).data[0],
                    torch.min(residual[~zm]).data[0],
                    torch.max(residual[~zm]).data[0]))
                gif(vis + 'pre_enc_residual', gif_prep(residual))
                gif(vis + 'pre_enc_output', gif_prep(stack))
            if use_preencoder == "only":
                # only run the preencoder and return the results
                return stack

        encodings = [self.enclist[0](stack)]
        for idx in range(1, self.size-self.topskips):
            encodings.append(
                self.enclist[idx](self.down(encodings[-1]), vis=vis))

        rdim = stack.shape[-2] // (2 ** (self.size - 1 - self.topskips))
        field_so_far = torch.zeros((1, rdim, rdim, 2),
                                   device=encodings[0].device)  # zero field
        residuals = []
        for i in range(self.size - 1 - self.topskips, target_level - 1, -1):
            if i >= self.skip:
                inputs_i = encodings[i]
                resampled_source = gridsample_residual(
                    inputs_i[:, 0:inputs_i.size(1)//2],
                    field_so_far, padding_mode='zeros')
                new_input_i = torch.cat(
                    (resampled_source, inputs_i[:, inputs_i.size(1)//2:]), 1)
                factor = (self.TRAIN_SIZE / (2. ** i)) / new_input_i.size()[-1]
                rfield = self.mlist[i](new_input_i) * factor
                residuals.append(rfield)
                # Resample field_so_far using rfield. Add rfield to the result
                # to produce the new field_so_far.
                resampled_field_so_far = gridsample_residual(
                    field_so_far.permute(0, 3, 1, 2), rfield,
                    padding_mode='border').permute(0, 2, 3, 1)
                field_so_far = rfield + resampled_field_so_far
            if i != target_level:
                field_so_far = self.up(
                    field_so_far.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return field_so_far, residuals


class PyramidTransformer(nn.Module):
    def __init__(self, size=4, dim=1536, skip=0, topskips=0, k=7, student=False,
                 num_targets=1, *args, **kwargs):
        super(PyramidTransformer, self).__init__()
        if not student:
            self.pyramid = EPyramid(size, dim, skip, topskips, k, num_targets)
        else:
            assert False  # TODO: add student network

    def open_layer(self):
        if self.pyramid.skip > 0:
            self.pyramid.skip -= 1
            print('Pyramid now using', self.pyramid.size - self.pyramid.skip,
                  'layers.')

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
        field, residuals = self.pyramid(x, idx, vis,
                                        use_preencoder=use_preencoder)
        return field

    ################################################################
    # Begin Sergiy API
    ################################################################

    @staticmethod
    def load(archive_path=None, height=5, dim=1536, skips=0, topskips=0, k=7,
             cuda=True, num_targets=1):
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

        model = PyramidTransformer(size=height, dim=dim, k=k, skip=skips,
                                   topskips=topskips, num_targets=num_targets)
        if cuda:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        model.train(False)

        print('Loading model state from {}...'.format(archive_path))
        state_dict = torch.load(archive_path)
        load_model_from_dict(model, state_dict)
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
        source = source.squeeze().unsqueeze(0)
        target = target.squeeze().unsqueeze(0)
        stack = torch.cat((source, target), 0).unsqueeze(0)
        return self(stack, idx=skip, vis=vis, use_preencoder=use_preencoder)
