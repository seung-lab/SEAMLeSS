import torch
import numpy as np
import skimage
from skimage import feature
from scipy.ndimage.measurements import label

from residuals import res_warp_img


def get_mse_and_smoothness_masks(src_var, tgt_var, pred_res_var, pred_tgt_var, masks, white_threshold):
    masks = masks[0]

    src_edge = (1 - masks[0:1])

    src_defects = (1 - masks[2:3]) * get_defect_mask(src_var, threshold=white_threshold)
    tgt_defects = (1 - masks[3:4]) * get_defect_mask(tgt_var, threshold=white_threshold)

    src_plastic = (1 - masks[4:5])
    tgt_plastic = (1 - masks[5:6])

    src_white = get_white_mask(src_var, threshold=white_threshold)
    tgt_white = get_white_mask(tgt_var, threshold=white_threshold)

    warped_src_edge = res_warp_img(src_edge.unsqueeze(0), pred_res_var,
                                   is_pix_res=True)

    warped_src_defects = res_warp_img(src_defects, pred_res_var,
                                      is_pix_res=True)

    warped_src_plastic = res_warp_img(src_plastic.unsqueeze(0), pred_res_var,
                                      is_pix_res=True)

    warped_src_white = res_warp_img(src_white, pred_res_var,
                                    is_pix_res=True)
    #warped_src_white = get_white_mask(pred_tgt_var)

    defect_mask = tgt_defects * warped_src_defects
    is_tissue_mask = tgt_white * warped_src_white #* tgt_plastic * warped_src_plastic

    mse_mask = defect_mask * warped_src_edge * is_tissue_mask
    smoothness_mask = defect_mask * warped_src_edge

    return mse_mask, smoothness_mask


def get_raw_defect_mask(img, threshold=-0.10):
    img_np = img.cpu().detach().numpy()
    result_np = np.logical_or((img_np > threshold), (img_np < -0.15))
    result = torch.FloatTensor(result_np.astype(int)).cuda()
    return result

def get_raw_white_mask(img):
    result = img >= -0.15
    return result.type(torch.FloatTensor).cuda()

def get_defect_mask(img, threshold=-3.5):
    img_np = img.cpu().detach().numpy()
    result_np = np.logical_or((img_np > threshold), (img_np < -3.99))
    result = torch.FloatTensor(result_np.astype(int)).cuda()
    return result


def get_white_mask(img, threshold=-3.5):
    result = img >= threshold
    return result.type(torch.FloatTensor).cuda()


def get_black_mask(img):
    result = img < 3.55
    return result.type(torch.FloatTensor).cuda()


# Numpy masks
def get_tissue_mask(img):
    mask = img < -0.15
    mask2 = filter_biggest_component(mask)
    return mask2


def filter_biggest_component(array):
    indices = np.indices(array.shape).T[:, :, [1,  0]]
    result = np.zeros_like(array)

    structure = np.ones((3, 3), dtype=np.int)
    structure[0, 0] = 0
    structure[0, -1] = 0
    structure[-1, 0] = 0
    structure[-1, -1] = 0

    labeled, ncomponents = label(array, structure)

    max_len = 0
    max_id  = None
    for i in range(ncomponents):
        my_guys = indices[labeled == i]

        if (len(my_guys) > max_len):
            coord = my_guys[0]
            if array[coord[0], coord[1]]:
                max_len = len(my_guys)
                max_id = i

    my_guys = indices[labeled == max_id]
    for coord in my_guys:
        result[coord[0], coord[1]] = True

    return result


def filter_connected_component(array, N):
    indices = np.indices(array.shape).T[:, :, [1, 0]]
    result = np.copy(array)
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(array, structure)
    for i in range(ncomponents):
        my_guys = indices[labeled == i]
        if (len(my_guys) < N):
            for coord in my_guys:
                result[coord[0], coord[1]] = False
    return result


def find_image_edge_np(img):
    img_black = get_tissue_mask(img)
    edges = feature.canny(img_black)

    edges[0:5, :] = False
    edges[:, 0:5] = False
    edges[-5:-1, :] = False
    edges[:, -5:-1] = False
    return coarsen_mask(edges1)


def coarsen_mask(mask):
    dmask = skimage.measure.block_reduce(mask, (2, 2), np.max)
    umask = dmask.repeat(2, axis=0).repeat(2, axis=1)
    return umask
