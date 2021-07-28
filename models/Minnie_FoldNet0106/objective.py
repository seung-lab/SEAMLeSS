import torch.nn as nn
import torch.nn.functional as F

class Objective(nn.Module):
    """
    Module that calculates an objective function on the net's outputs.

    `Objective` is implemented as a PyTorch module for ease of distributing
    the calculation across GPUs with DataParallel, and for modularity.
    It is bundled with the archive because models can often have
    different objective functions, the calculation of which can
    depend on the model's specific architecture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, preds, sample):
        losses = dict()

        ## Discrim
        for k in len(preds):
        
            loss = F.binary_cross_entropy_with_logits(input=preds[k], target=sample[k])
            losses[k] = loss.unsqueeze(0)
            
        return losses


ValidationObjective = Objective
