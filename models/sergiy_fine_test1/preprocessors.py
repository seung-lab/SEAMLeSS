import torch

class TissueNormalizer(object):
    def __init__(self, black_threshold=-0.4, black_fill=0):
        self.normer = torch.nn.InstanceNorm1d(1)
        self.black_fill = black_fill
        self.black_threshold = black_threshold

    def __call__(self, x, defect_mask):
        assert x.shape[0] == 1

        if defect_mask is not None:
            assert defect_mask.shape[0] == 1

        for channel in range(x.shape[1]):
            img = x[0, channel]
            # plastic mask is 1 at plastic and 0 elsewhere
            # we want to ignore plastic
            is_black = get_threshold_black(img, self.black_threshold)
            tissue = (1 - is_black)
            if defect_mask is not None:
                tissue = tissue * (1 - defect_mask[0, channel])

            good_mask = tissue >= 0.9
            bad_mask  = tissue < 0.9

            if img[good_mask].shape[0] > 0:
                img[good_mask] = self.normer(img[good_mask].unsqueeze(0).unsqueeze(0)).squeeze()
            img[bad_mask]  = self.black_fill

        return x

class RangeAdjuster(object):
    def __init__(self, divide=255.0, subtract=0.5):
        self.divide = divide
        self.subtract = subtract

    def __call__(self, x):
        return (x / self.divide) - self.subtract

def get_threshold_black(img, threshold=-0.1):
    return (img < threshold).type(torch.cuda.FloatTensor)
