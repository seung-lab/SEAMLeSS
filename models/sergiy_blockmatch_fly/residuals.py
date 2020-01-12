from utilities.helpers import grid_sample
import copy
import torch

def res_warp_res(res_a, res_b, is_pix_res=True):
    if is_pix_res:
        res_b = 2 * res_b / (res_b.shape[-2])

    if len(res_a.shape) == 4:
        result = grid_sample(
                        res_a.permute(0, 3, 1, 2),
                        res_b,
                        padding_mode='border').permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = grid_sample(
                        res_a.permute(2, 0, 1).unsqueeze(0),
                        res_b.unsqueeze(0),
                        padding_mode='border')[0].permute(1, 2, 0)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")

    return result


def res_warp_img(img, res_in, is_pix_res=True, padding_mode='zeros'):

    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in

    if len(img.shape) == 4:
        result = grid_sample(img, res, padding_mode=padding_mode)
    elif len(img.shape) == 3:
        if len(res.shape) == 3:
            result = grid_sample(img.unsqueeze(0),
                                         res.unsqueeze(0), padding_mode=padding_mode)[0]
        else:
            img = img.unsqueeze(1)
            result = grid_sample(img,
                                         res, padding_mode=padding_mode).squeeze(1)
    elif len(img.shape) == 2:
        result = grid_sample(img.unsqueeze(0).unsqueeze(0),
                                     res.unsqueeze(0),
                                     padding_mode=padding_mode)[0, 0]
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    return result


def combine_residuals(a, b, is_pix_res=True):
    return b + res_warp_res(a, b, is_pix_res=is_pix_res)

upsampler = torch.nn.UpsamplingBilinear2d(scale_factor=2)
def upsample_residuals(residuals):
    result = upsampler(residuals.permute(
                                     0, 3, 1, 2)).permute(0, 2, 3, 1)
    result *= 2
    return result

def downsample_residuals(residuals):
    result = torch.nn.functional.avg_pool2d(residuals.permute(
                                     0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    result /= 2
    return result

