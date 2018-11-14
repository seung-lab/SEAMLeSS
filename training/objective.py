import torch
import torch.nn as nn
from loss import smoothness_penalty
from utilities.helpers import gridsample_residual


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
    It is bundled with the archive because models can often have
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
