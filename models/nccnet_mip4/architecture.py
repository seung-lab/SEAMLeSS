import torch
import torch.nn as nn
from nccnet import KitModel


class Model(nn.Module):
    """
    This is a wrapper class for the architecture code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encode = None
        self.forward = self.encode.forward

    def __getitem__(self, index):
        return self.submodule(index)

    def __len__(self):
        return self.height

    def load(self, path):
        """
        Loads saved weights into the model
        """
        self.encode = KitModel(path)
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        with path.open('wb') as f:
            torch.save(self.encode.state_dict(), f)

    @property
    def height(self):
        return 1

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
        return self

    def train_level(self, *args, **kwargs):
        return self

    def init_level(self, *args, **kwargs):
        return self
