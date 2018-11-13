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
            X[..., i, :, :] = self.contrast(X[..., i, :, :], mask=mask)
        return X

    def contrast(self, X, mask=...):
        """
        Performs contrast Limited Adaptive Histogram Equalization
        """
        Xb = (X * 255).to(torch.uint8)[mask].squeeze().numpy()
        eq = self.clahe.apply(Xb)
        X[mask] = torch.from_numpy(eq).unsqueeze(0).to(torch.float) / 255
        X[Xb == 0] = 0

    def gen_mask(self, X, threshold=1):
        return ...
