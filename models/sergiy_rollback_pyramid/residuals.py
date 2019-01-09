from .helpers import gridsample_residual


def res_warp_res(res_a, res_b, is_pix_res=True):
    if is_pix_res:
        res_b = 2 * res_b / (res_b.shape[0])

    if len(res_a.shape) == 4:
        result = gridsample_residual(
                        res_a.permute(0, 3, 1, 2),
                        res_b,
                        padding_mode='border').permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = gridsample_residual(
                        res_a.permute(2, 0, 1).unsqueeze(0),
                        res_b.unsqueeze(0),
                        padding_mode='border')[0].permute(1, 2, 0)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")

    return result


def res_warp_img(img, res_in, is_pix_res=True):

    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in

    if len(img.shape) == 4:
        result = gridsample_residual(img, res, padding_mode='zeros')
    elif len(img.shape) == 3:
        result = gridsample_residual(img.unsqueeze(0),
                                     res.unsqueeze(0), padding_mode='zeros')[0]
    elif len(img.shape) == 2:
        result = gridsample_residual(img.unsqueeze(0).unsqueeze(0),
                                     res.unsqueeze(0),
                                     padding_mode='zeros')[0, 0]
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    return result


def combine_residuals(a, b, is_pix_res=True):
    return res_warp_res(a, b, is_pix_res=is_pix_res) + b


