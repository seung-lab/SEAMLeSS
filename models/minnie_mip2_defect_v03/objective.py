import torch.nn as nn


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

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, prediction, cracks, folds):  # TODO: use masks
        cracks = cracks.to(prediction.device)
        folds = folds.to(prediction.device)
        return ((prediction[:, 0:1] - cracks) ** 2
                + (prediction[:, 1:2] - folds) ** 2).mean()


ValidationObjective = Objective
