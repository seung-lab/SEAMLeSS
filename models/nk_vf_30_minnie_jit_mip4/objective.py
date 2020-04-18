import torch
import torch.nn as nn
import torchfields # noqa: unused
from utilities import masklib
from training.loss import smoothness_penalty


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
        kwargs['supervised'] = False
        super().__init__(*args, **kwargs)


class SupervisedLoss(nn.Module):
    """
    Calculates a supervised loss based on the mean squared error with
    the ground truth vector field.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, sample, prediction):  # TODO: use masks
        truth = sample.truth.to(prediction.device)
        return ((prediction - truth) ** 2).mean() * 25000


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
        self.eps = 1e-7

    def forward(self, sample, prediction, tgt_field=None):
        prediction.field_()
        masks = prepare_masks(sample)
        src_masks = masks['src_masks']
        tgt_masks = masks['tgt_masks']
        src_field_masks = masks['src_field_masks']
        tgt_field_masks = masks['tgt_field_masks']

        src = sample.src.image.to(prediction.device)
        tgt = sample.tgt.image.to(prediction.device)

        if tgt_field is not None:
            tgt_field = tgt_field.permute(0,3,1,2)
            tgt_field.field_()
            tgt = tgt_field.sample(tgt)
            tgt_masks = [tgt_field.sample(m) for m in tgt_masks]
            tgt_field_masks = [tgt_field.sample(m) for m in tgt_field_masks]

        src_warped = prediction.sample(src)
        image_loss_map = (src_warped - tgt)**2
        if src_masks or tgt_masks:
            image_weights = torch.ones_like(image_loss_map)
            if src_masks is not None:
                for mask in src_masks:
                    try:
                        mask = prediction.sample(mask)
                        image_loss_map = image_loss_map * mask
                        image_weights = image_weights * mask
                    except Exception as e:
                        print('Source mask failed: {}: {}'.format(e.__class__.__name__, e))
                        print('mask shape:', mask.shape)
            if tgt_masks is not None:
                for mask in tgt_masks:
                    try:
                        image_loss_map = image_loss_map * mask
                        image_weights = image_weights * mask
                    except Exception as e:
                        print('Target mask failed: {}: {}'.format(e.__class__.__name__, e))
                        print('mask shape:', mask.shape)
            mse_loss = image_loss_map.sum() / (image_weights.sum() + self.eps)
        else:
            mse_loss = image_loss_map.mean()
        sample.image_loss_map = image_loss_map

        field_loss_map = self.field_penalty([prediction.permute(0, 2, 3, 1)])
        if src_field_masks or tgt_field_masks:
            field_weights = torch.ones_like(field_loss_map)
            if src_field_masks is not None:
                for mask in src_field_masks:
                    try:
                        mask = prediction.sample(mask)
                        field_loss_map = field_loss_map * mask
                        field_weights = field_weights * mask
                    except Exception as e:
                        print('Source field mask failed: {}: {}'.format(e.__class__.__name__, e))
                        print('mask shape:', mask.shape)
            if tgt_field_masks is not None:
                for mask in tgt_field_masks:
                    try:
                        field_loss_map = field_loss_map * mask
                        field_weights = field_weights * mask
                    except Exception as e:
                        print('Target field mask failed: {}: {}'.format(e.__class__.__name__, e))
                        print('mask shape:', mask.shape)
            field_loss = field_loss_map.sum() / field_weights.sum()
        else:
            field_loss = field_loss_map.mean()
        sample.field_loss_map = field_loss_map

        loss = (mse_loss + self.lambda1 * field_loss)
        return {
            'loss': loss,
            'mse_loss': mse_loss,
            'smooth_loss': self.lambda1 * field_loss
        }

@torch.no_grad()
def gen_masks(src, tgt, threshold=0):
    """
    Heuristic generation of masks based on simple thresholding.
    Can be expanded in the future to include more sophisticated methods.

    A better option is to include the masks as part of the dataset and pass
    them to prepare_masks().
    """
    src, tgt = (src * 255).to(torch.uint8), (tgt * 255).to(torch.uint8)
    return (src < threshold), (tgt < threshold)


@torch.no_grad()
def prepare_masks(sample, threshold=127):
    """
    Returns properly formatted masks with which to weight the loss function.
    If masks is None, this calls gen_masks to generate them.
    """
    # MSE coefficient on the folds (src, tgt) and radius:
    fold_main_mse_coef = 0, 0
    fold_main_mse_radius = 0

    # MSE coefficient in the fold neighborhood (src, tgt) and radius:
    fold_surround_mse_coef = 1.0, 1.0
    fold_surround_mse_radius = 0

    # MSE coefficient on the cracks (src, tgt) and radius:
    crack_main_mse_coef = 0.5, 0.0
    crack_main_mse_radius = 0

    # MSE coefficient in the crack neighborhood (src, tgt) and radius:
    crack_surround_mse_coef = 1.0, 1.0
    crack_surround_mse_radius = 0

    # smoothness coefficient on the folds (src, tgt) and radius:
    fold_main_field_coef = 0.000001, 1.0
    fold_main_field_radius = 0

    # smoothness coefficient in the fold neighborhood (src, tgt) and radius:
    fold_surround_field_coef = 1.0, 1.0
    fold_surround_field_radius = 0

    # smoothness coefficient on the cracks (src, tgt) and radius:
    crack_main_field_coef = 0.00000001, 1.0
    crack_main_field_radius = 0

    # smoothness coefficient in the crack neighborhood (src, tgt) and radius:
    crack_surround_field_coef = 0.25, 1.0
    crack_surround_field_radius = 1


    src, tgt = sample.src.image, sample.tgt.image
    src_weights, tgt_weights = torch.ones_like(src), torch.ones_like(tgt)
    src_field_weights = torch.ones_like(src)
    tgt_field_weights = torch.ones_like(tgt)

    if sample.src.folds is None or len(sample.src.folds) == 0:
        assert False
    if sample.src.cracks is None or len(sample.src.cracks) == 0:
        assert False

    src_fold_defects = sample.src.folds > threshold
    tgt_fold_defects = sample.tgt.folds > threshold
    src_crack_defects = sample.src.cracks > threshold
    tgt_crack_defects = sample.tgt.cracks > threshold

    # Fold (MSE) masks
    src_fold_main = masklib.dilate(src_fold_defects, radius=fold_main_mse_radius)
    tgt_fold_main = masklib.dilate(tgt_fold_defects, radius=fold_main_mse_radius)
    src_fold_surround = masklib.dilate(src_fold_defects, radius=fold_surround_mse_radius)
    tgt_fold_surround = masklib.dilate(tgt_fold_defects, radius=fold_surround_mse_radius)

    # Crack (MSE) masks
    src_crack_main = masklib.dilate(src_crack_defects, radius=crack_main_mse_radius)
    tgt_crack_main = masklib.dilate(tgt_crack_defects, radius=crack_main_mse_radius)
    src_crack_surround = masklib.dilate(src_crack_defects, radius=crack_surround_mse_radius)
    tgt_crack_surround = masklib.dilate(tgt_crack_defects, radius=crack_surround_mse_radius)

    # coefficient in the fold/crack neighborhood:
    src_weights[src_fold_surround], tgt_weights[tgt_fold_surround] = fold_surround_mse_coef
    src_weights[src_crack_surround], tgt_weights[tgt_crack_surround] = crack_surround_mse_coef

    # coefficient on the fold/crack:
    src_weights[src_fold_main], tgt_weights[tgt_fold_main] = fold_main_mse_coef
    src_weights[src_crack_main], tgt_weights[tgt_crack_main] = crack_main_mse_coef

    # no MSE outside tissue
    src_weights[(src*255.0).to(torch.uint8) < 1] = 0
    tgt_weights[(tgt*255.0).to(torch.uint8) < 1] = 0


    # Fold (Field) masks
    src_fold_field_main = masklib.dilate(src_fold_defects, radius=fold_main_field_radius)
    tgt_fold_field_main = masklib.dilate(tgt_fold_defects, radius=fold_main_field_radius)
    src_fold_field_surround = masklib.dilate(src_fold_defects, radius=fold_surround_field_radius)
    tgt_fold_field_surround = masklib.dilate(tgt_fold_defects, radius=fold_surround_field_radius)

    # Crack (Field) masks
    src_crack_field_main = masklib.dilate(src_crack_defects, radius=crack_main_field_radius)
    tgt_crack_field_main = masklib.dilate(tgt_crack_defects, radius=crack_main_field_radius)
    src_crack_field_surround = masklib.dilate(src_crack_defects, radius=crack_surround_field_radius)
    tgt_crack_field_surround = masklib.dilate(tgt_crack_defects, radius=crack_surround_field_radius)

    # coefficient in the fold/crack neighborhood:
    src_field_weights[src_fold_field_surround], tgt_field_weights[tgt_fold_field_surround] = fold_surround_field_coef
    src_field_weights[src_crack_field_surround], tgt_field_weights[tgt_crack_field_surround] = crack_surround_field_coef

    # coefficient on the fold/crack:
    src_field_weights[src_fold_field_main], tgt_field_weights[tgt_fold_field_main] = fold_main_field_coef
    src_field_weights[src_crack_field_main], tgt_field_weights[tgt_crack_field_main] = crack_main_field_coef

    # less strict field outside src tissue (to extend folds)
    src_field_weights[(src*255.0).to(torch.uint8) < 1] = 0.001

    src_aug_masks = ([1.0 - m.float() for m in sample.src.aug_masks]
                     if sample.src.aug_masks else [])
    tgt_aug_masks = ([1.0 - m.float() for m in sample.tgt.aug_masks]
                     if sample.tgt.aug_masks else [])

    return {'src_masks': [src_weights.float()] + src_aug_masks,
            'tgt_masks': [tgt_weights.float()] + tgt_aug_masks,
            'src_field_masks': [src_field_weights.float()],
            'tgt_field_masks': [tgt_field_weights.float()]}
