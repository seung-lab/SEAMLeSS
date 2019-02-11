import torch
import torch.nn as nn
import cv2


class Preprocessor(nn.Module):
    """
    Code to preprocess the data.
    This can include contrast normalization, masking, etc.

    While this does necessarily not need to be a PyTorch module, it inherits
    from nn.Module to make it easier to parallelize with DataParallel if
    desired.
    """

    def __init__(self, clipLimit=40, tileGridSize=(8, 8), *args, **kwargs):
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                     tileGridSize=tileGridSize)

    def forward(self, X, *args, **kwargs):
        for i in range(X.shape[-3]):
            mask = self.gen_mask(X[..., i, :, :])
            X[..., i, :, :] = self.normalize(X[..., i, :, :], mask=mask)
            X[..., i, :, :] = self.contrast(X[..., i, :, :], mask=mask)
        return X

    def contrast(self, X, mask=...):
        """
        Performs Contrast Limited Adaptive Histogram Equalization
        """
        Xb = (X * 255).to(torch.uint8)[mask].squeeze()
        eq = torch.from_numpy(self.clahe.apply(Xb.cpu().numpy()))
        X[mask] = eq.unsqueeze(0).to(torch.float) / 255
        X[..., Xb == 0] = 0
        return X

    def normalize(self, X, mask=..., min=0, max=1):
        """
        Rescale values from min to max
        """
        X[mask] = X[mask] - X[mask].min()
        X[mask] = X[mask] / X[mask].max() * (max - min) + min
        return X

    def gen_mask(self, X, threshold=1):
        """
        Return a mask that selects only the relevant parts of the image
        """
        return ...
