import torch
import torch.nn as nn
import copy
from helpers import gridsample_residual, upsample, downsample, load_model_from_dict


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, height, feature_maps=12, *args, **kwargs):
        super().__init__()
        self.height = height
        self.encode = EncodingPyramid([1] + [feature_maps]*(height - 1))
        self.align = AligningPyramid(height)

    def __getitem__(self, index):
        return self.submodule(index)

    def forward(self, src, tgt, in_field=None):
        src, tgt = self.encode(src, tgt)
        field = self.align(src, tgt, in_field)
        return field

    def load(self, path):
        """
        Loads saved weights into the model
        """
        with path.open('rb') as f:
            weights = torch.load(f)
        load_model_from_dict(self, weights)
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        with path.open('wb') as f:
            torch.save(self.state_dict(), f)

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
        if index is None or (isinstance(index, int)
                             and index >= self.height):
            index = slice(self.height)
        return _SubmoduleView(self, index)


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


###################
# Loss calculation
###################

class Objective(nn.Module):
    """
    Module that calculates an objective function on the net's outputs.

    `supervised` indicates the type of loss to be used: supervised if True,
    self-supervised if False. Other possible values can be added.

    `function` is an optional argument that, if specified, overrides this
    with a custom function. It must be a callable and differentiable,
    accept inputs in the format of the outputs produced by the model,
    and produce a PyTorch scalar as output.

    `Objective` is implemented as a PyTorch module for ease of distributing
    the calculation across GPUs with DataParallel, and for modularity.
    It is bundled with the architecture code because models can often have
    different objective functions, the calculation of which can
    depend on the model's specific architecture.
    """

    def __init__(self, *args, supervised=True, function=None, **kwargs):
        super().__init__()
        if function is not None:
            if callable(function):
                self.function = function
            else:
                raise TypeError('Cannot use {} as an objective function. '
                                'Must be a callable.'.format(type(function)))
        elif supervised:
            self.function = SupervisedLoss(*args, **kwargs)
        else:
            self.function = SelfSupervisedLoss(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class ValidationObjective(Objective):
    """
    Calculates a validation objective function on the net's outputs.

    This is currently set to simply be the self-supervised loss,
    but this could be changed here to Pearson correlation or some
    other measure without affecting the training objective.
    """

    def __init__(self, *args, **kwargs):
        if 'supervised' in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != 'supervised'}
        super().__init__(*args, supervised=False, **kwargs)


class SupervisedLoss(nn.Module):
    """
    Calculates a supervised loss based on the mean squared error with
    the ground truth vector field.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, prediction, truth, **masks):  # TODO: use masks
        truth = truth.to(prediction.device)
        return ((prediction - truth) ** 2).mean()


class SelfSupervisedLoss(nn.Module):
    """
    Calculates a self-supervised loss based on
    (a) the mean squared error between the source and target images
    (b) the smoothness of the vector field

    The masks are used to ignore or reduce the loss values in certain regions
    of the images and vector field.

    If `MSE(a, b)` is the mean squared error of two images, and `Penalty(f)`
    is the smoothness penalty of a vector field, the loss is calculated
    roughly as
        >>> loss = MSE(src, tgt) + lambda1 * Penalty(prediction)
    """

    def __init__(self, penalty, lambda1, *args, **kwargs):
        super().__init__()
        from loss import smoothness_penalty  # TODO: clean up
        self.field_penalty = smoothness_penalty(penalty)
        self.lambda1 = lambda1

    def forward(self, src, tgt, prediction,
                src_masks=None, tgt_masks=None,
                src_field_masks=None, tgt_field_masks=None):
        src, tgt = src.to(prediction.device), tgt.to(prediction.device)

        src_warped = gridsample_residual(src, prediction, padding_mode='zeros')
        image_loss_map = (src_warped - tgt)**2
        if src_masks or tgt_masks:
            image_weights = torch.ones_like(image_loss_map)
            if src_masks is not None:
                for mask in src_masks:
                    mask = gridsample_residual(mask, prediction,
                                               padding_mode='border')
                    image_loss_map = image_loss_map * mask
                    image_weights = image_weights * mask
            if tgt_masks is not None:
                for mask in tgt_masks:
                    image_loss_map = image_loss_map * mask
                    image_weights = image_weights * mask
            mse_loss = image_loss_map.sum() / image_weights.sum()
        else:
            mse_loss = image_loss_map.mean()

        field_loss_map = self.field_penalty([prediction])
        if src_field_masks or tgt_field_masks:
            field_weights = torch.ones_like(field_loss_map)
            if src_field_masks is not None:
                for mask in src_field_masks:
                    mask = gridsample_residual(mask, prediction,
                                               padding_mode='border')
                    field_loss_map = field_loss_map * mask
                    field_weights = field_weights * mask
            if tgt_field_masks is not None:
                for mask in tgt_field_masks:
                    field_loss_map = field_loss_map * mask
                    field_weights = field_weights * mask
            field_loss = field_loss_map.sum() / field_weights.sum()
        else:
            field_loss = field_loss_map.mean()

        loss = (mse_loss + self.lambda1 * field_loss) / 25000
        return loss


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
