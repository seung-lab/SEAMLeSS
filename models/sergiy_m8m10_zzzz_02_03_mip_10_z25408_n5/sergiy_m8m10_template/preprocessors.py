import torch

class TissueNormalizer(object):
    def __init__(self, black_value=-4.0):
        self.normer = torch.nn.InstanceNorm1d(1)
        self.black_value = black_value

    def __call__(self, x, plastic_mask):
        assert x.shape[0] == 1
        if plastic_mask:
            assert plastic_mask.shape[0] == 1

        for channel in [0, 1]:
            img = x[0, channel]
            # plastic mask is 1 at plastic and 0 elsewhere
            # we want to ignore plastic
            is_black = get_threshold_black(img)
            tissue = (1 - is_black)
            if plastic_mask:
                is_plastic = plastic_mask[0, channel]
                tissue = tissue * (1 - is_plastic)

            good_mask = tissue >= 0.99
            bad_mask  = tissue < 0.99

            if img[good_mask].shape[0] > 0:
                img[good_mask] = self.normer(img[good_mask].unsqueeze(0).unsqueeze(0)).squeeze()
            img[bad_mask]  = self.black_value

        return x

class RangeAdjuster(object):
    def __init__(self, divide=255.0, subtract=0.5):
        self.divide = divide
        self.subtract = subtract

    def __call__(self, x):
        return (x / self.divide) - self.subtract

def get_threshold_black(img, threshold=-0.1):
    return (img < threshold).type(torch.cuda.FloatTensor)
