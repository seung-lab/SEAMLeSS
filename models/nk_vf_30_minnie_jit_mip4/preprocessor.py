import torch
import torch.nn as nn
import cv2
import numpy as np

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
            X[..., i, :, :] = self.contrast(X[..., i, :, :])
            X[..., i, :, :] = self.normalize(X[..., i, :, :], mask=mask, min=1.0/255.0)
        return X

    def contrast(self, X, mask=...):
        """
        Performs Contrast Limited Adaptive Histogram Equalization
        """
        Xb = (X * 255).to(torch.uint8).squeeze()
        Xb_np = Xb.cpu().numpy()
        Xb_0 = Xb_np == 0
        try:
            Xb_np[Xb_0] = int(np.mean(Xb_np[Xb_np != 0]))
        except ValueError:
            pass

        res = self.clahe.apply(Xb_np)
        res[Xb_0] = 0
        eq = torch.from_numpy(res)

        X[...] = eq.unsqueeze(0).to(torch.float) / 255
        return X

    def normalize(self, X, mask=..., min=0, max=1):
        """
        Rescale values from min to max
        """
        if len(X[mask]) > 0 and X[mask].max() > 0.0:
            X[mask] = X[mask] - X[mask].min()
            X[mask] = X[mask] / X[mask].max() * (max - min) + min
        else:
            X[mask] = min

        return X

    def gen_mask(self, X, threshold=1):
        """
        Return a mask that selects only the relevant parts of the image
        """
        return X > 0.0
