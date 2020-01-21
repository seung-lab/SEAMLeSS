from utilities.helpers import grid_sample
import copy
import torch
import torchfields

def shift_by_int(img, x_shift, y_shift, is_res=False):
    if is_res:
        img = img.permute(0, 3, 1, 2)

    x_shifted = torch.zeros_like(img)
    if x_shift > 0:
        x_shifted[..., x_shift:, :]  = img[..., :-x_shift, :]
    elif x_shift < 0:
        x_shifted[..., :x_shift, :]  = img[..., -x_shift:, :]
    else:
        x_shifted = img.clone()

    result = torch.zeros_like(img)
    if y_shift > 0:
        result[..., y_shift:]  = x_shifted[..., :-y_shift]
    elif y_shift < 0:
        result[..., :y_shift]  = x_shifted[..., -y_shift:]
    else:
        result = x_shifted.clone()

    if is_res:
        result = result.permute(0, 2, 3, 1)

    return result




def upsample(x, is_res=False, is_pix_res=True, mode='nearest'):
    if is_res:
        x = x.permute(0, 3, 1, 2)
        print (x.abs().mean() * 100)
    print (mode)
    result = torch.nn.functional.interpolate(x, scale_factor=2, mode=mode)

    if is_res:
        result = result.permute(0, 2, 3, 1)

    if is_res and is_pix_res:
        result *= 2
        print (result.abs().mean() * 100)
    return result

def downsample(x, is_res=False, is_pix_res=True):
    if is_res:
        x = x.permute(0, 3, 1, 2)
    downsampler = torch.nn.AvgPool2d(2)
    result = downsampler(x)
    if is_res:
        result = result.permute(0, 2, 3, 1)
    if is_res and is_pix_res:
        result /= 2
    return result


def res_warp_res(res_a, res_b, is_pix_res=True, rollback=0):
    if len(res_a.shape) == 4:
        res_a_img = res_a.permute(0, 3, 1, 2)
    elif len(res_a.shape) == 3:
        res_a_img = res_a.permute(2, 0, 1)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")
    result_perm = res_warp_img(res_a_img, res_b, is_pix_res, rollback)

    if len(res_a.shape) == 4:
        result = result_perm.permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = result_perm.permute(1, 2, 0)

    return result


def res_warp_img(img, res_in, is_pix_res=True, rollback=0):
    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in
    original_shape = copy.deepcopy(img.shape)

    if len(img.shape) == 4:
        img_unsq = img
        res_unsq = res
    elif len(img.shape) == 3:
        img_unsq = img.unsqueeze(0)
        res_unsq = res.unsqueeze(0)
    elif len(img.shape) == 2:
        img_unsq = img.unsqueeze(0).unsqueeze(0)
        res_unsq = res.unsqueeze(0)
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    img_unsq_rollb = img_unsq
    res_unsq_rollb = res_unsq
    for i in range(rollback):
        img_unsq_rollb = upsample(img_unsq_rollb)
        res_unsq_rollb = upsample(res_unsq_rollb, is_res=True, is_pix_res=False)
    result_unsq_rollb = grid_sample(img_unsq_rollb, res_unsq_rollb, padding_mode='zeros')

    result_unsq = result_unsq_rollb
    for i in range(rollback):
        result_unsq = downsample(result_unsq)

    result = result_unsq
    while len(result.shape) > len(original_shape):
        result = result.squeeze(0)

    return result


def combine_residuals(a, b, is_pix_res=True, rollback=0):
    return res_warp_res(a, b, is_pix_res=is_pix_res, rollback=rollback) + b


