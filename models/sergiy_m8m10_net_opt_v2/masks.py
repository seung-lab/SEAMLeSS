import torch
import numpy as np
import skimage
import h5py
import time
import six

from skimage import feature
from scipy.ndimage import convolve
from scipy.ndimage.measurements import label
from .residuals import res_warp_img

from pdb import set_trace as st

def find_image_edge(img):
    img_black = get_tissue_mask(img)
    edges1 = feature.canny(img_black)

    edges1[0:5, :] = False
    edges1[:, 0:5] = False
    edges1[-5:-1, :] = False
    edges1[:, -5:-1] = False
    return edges1

def get_src_defect_mask(masks_var, reverse=True):
    if reverse:
        return masks_var[2]
    else:
        return masks_var[3]

def get_src_plastic_mask(masks_var, reverse=True):
    if reverse:
        return masks_var[4]
    else:
        return masks_var[5]

RAW_WHITE_THRESHOLD = -0.485
def get_raw_defect_mask(img, threshold=-1):
    #img_np = img.cpu().detach().numpy()
    #result_np = np.logical_or((img_np > threshold), (img_np < -0.15))
    #result = torch.FloatTensor(result_np.astype(int)).cuda()
    result = 1 - ((img < threshold) * (img > RAW_WHITE_THRESHOLD))
    return result.to(device=img.device, dtype=torch.float32)

def get_raw_white_mask(img):
    result = img >= RAW_WHITE_THRESHOLD
    return result.to(device=img.device, dtype=torch.float32)

def get_defect_mask(img, threshold=-3.5):
    #img_np = img.cpu().detach().numpy()
    #result_np = np.logical_or((img_np > threshold), (img_np < -3.99))
    #result = torch.FloatTensor(result_np.astype(int)).cuda()
    result = 1 - ((img < threshold) * (img > -3.9999))
    return result.to(device=img.device, dtype=torch.float32)

def get_brightness_mask(img, low_cutoff, high_cutoff):
    result = (img >= low_cutoff)* (img <= high_cutoff)
    return result.to(device=img.device, dtype=torch.float32)

def get_blood_vessel_mask(img, threshold=2.5):
    result = img >= threshold
    return result.to(device=img.device, dtype=torch.float32)

def get_white_mask(img, threshold=-3.5):
    result = img >= threshold
    return result.to(device=img.device, dtype=torch.float32)


def get_black_mask(img):
    result = img < 3.55
    return result.to(device=img.device, dtype=torch.float32)


# Numpy masks
def get_very_white_mask(img):
    # expects each pixel in range [-0.5, 0.5]
    # used at mip 8
    return img > 0.04


def get_plastic_mask(img):
    return filter_biggest_component(get_very_white_mask(img))



def filter_biggest_component(array, debug=False, only_positive=True):
    indices = np.indices(array.shape).T[:, :, [1,  0]]
    result = np.zeros_like(array)

    structure = np.ones((3, 3), dtype=np.int)
    structure[0, 0] = 0
    structure[0, -1] = 0
    structure[-1, 0] = 0
    structure[-1, -1] = 0

    labeled, ncomponents = label(array, structure)
    if debug:
        st()

    max_len = 0
    max_id  = None
    component_size = {}
    print (only_positive)
    for i in range(ncomponents):
        my_guys = indices[labeled == i]

        component_size[i] = len(my_guys)
        if (len(my_guys) > max_len):
            coord = my_guys[0]
            if (not only_positive) or array[coord[0], coord[1]]:
                max_len = len(my_guys)
                max_id = i
                print (max_len)
    my_guys = indices[labeled == max_id]
    for coord in my_guys:
        result[coord[0], coord[1]] = True

    return result


def filter_connected_component(array, N):
    indices = np.indices(array.shape).T[:, :, [1, 0]]
    result = np.copy(array)
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(array, structure)
    for i in range(1, ncomponents + 1):
        my_guys = indices[labeled == i]
        #print ("Component {}: {}".format(i, len(my_guys)))
        t, b = my_guys[0][0], my_guys[0][0]
        l, r = my_guys[0][1], my_guys[0][1]
        for coord in my_guys:
            b = min(b, coord[0])
            t = max(t, coord[0])
            l = min(l, coord[1])
            r = max(r, coord[1])
        size = (t - b) + (r - l)
        #print ("Size: {}".format((t - b) + (r - l)))
        if (size < N):
            #print ('filtered')
            for coord in my_guys:
                result[coord[0], coord[1]] = False
                b = min(b, coord[0])
                t = max(t, coord[0])
                l = min(l, coord[1])
                r = max(r, coord[1])
    return result

def connected_component(array):
    indices = np.indices(array.shape).T[:, :, [1, 0]]
    result = np.copy(array)
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(array, structure)
    print ("NCOMPONENTS: {}".format(ncomponents))
    for i in range(0, ncomponents + 1):
        my_guys = indices[labeled == i]
        #print ("Component {}: {}".format(i, len(my_guys)))
        t, b = my_guys[0][0], my_guys[0][0]
        l, r = my_guys[0][1], my_guys[0][1]
        for coord in my_guys:
            b = min(b, coord[0])
            t = max(t, coord[0])
            l = min(l, coord[1])
            r = max(r, coord[1])
            result[coord[0], coord[1]] = i
            #print ('marked as {}'.format(i))
        #print ("Size: {}".format((t - b) + (r - l)))
    return result

def find_image_edge_np(img):
    img_black = get_tissue_mask(img)
    edges = feature.canny(img_black)

    edges[0:5, :] = False
    edges[:, 0:5] = False
    edges[-5:-1, :] = False
    edges[:, -5:-1] = False
    return coarsen_mask(edges1)


def find_image_edge_np(img):
    img_black = get_tissue_mask(img)
    edges = feature.canny(img_black)

    edges[0:5, :] = False
    edges[:, 0:5] = False
    edges[-5:-1, :] = False
    edges[:, -5:-1] = False
    return coarsen_mask(edges1)

def coarsen_mask_pool(mask, n=1, flip=True):
    mask = mask.to(dtype=torch.float32)
    if flip:
        mask = 1 - mask

    mask_down = torch.nn.functional.max_pool2d(mask, n)
    mask_up   = torch.nn.functional.interpolate(mask_down.unsqueeze(0), scale_factor=n).squeeze(0)
    if flip:
        mask_up = 1 - mask_up
    return mask_up


def coarsen_mask(mask, n=1, flip=True):
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    for _ in range(n):
        if isinstance(mask, np.ndarray):
            mask = convolve(mask, kernel) > 0
            mask = mask.astype(np.int16) > 1
        else:
            mask = mask.to(dtype=torch.float32)
            kernel_var = torch.FloatTensor(kernel).to(device=mask.device).unsqueeze(0).unsqueeze(0)
            k = torch.nn.Parameter(data=kernel_var, requires_grad=False)
            if flip:
                mask = 1 - mask
            mask = (torch.nn.functional.conv2d(mask.unsqueeze(0),
                kernel_var, padding=1) > 1).squeeze(0).to(dtype=torch.float32, device=mask.device)
            if flip:
                mask = 1 - mask
    return mask

def get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply):
    result = torch.ones_like(bundle['src'], device=bundle['src'].device)
    pred_res = bundle['pred_res']
    src_mask_mse = get_warped_mask_set(bundle, pred_res,
                                   mse_keys_to_apply['src'])
    tgt_mask_mse = get_warped_mask_set(bundle, torch.zeros_like(pred_res),
                                   mse_keys_to_apply['tgt'])
    mask_mse = src_mask_mse * tgt_mask_mse

    src_mask_sm = get_warped_mask_set(bundle, pred_res,
                                   sm_keys_to_apply['src'])
    tgt_mask_sm = get_warped_mask_set(bundle, torch.zeros_like(pred_res),
                                   sm_keys_to_apply['tgt'])
    mask_sm = src_mask_sm * tgt_mask_sm

    return mask_mse, mask_sm

def binarize(img, bin_setting):
    if bin_setting['strat']== 'value':
        result = (img == bin_setting['value'])
    elif bin_setting['strat']== 'ne':
        result = (img != bin_setting['value'])
    elif bin_setting['strat']== 'eq':
        result = (img == bin_setting['value'])
    elif bin_setting['strat']== 'lt':
        result = (img < bin_setting['value'])
    elif bin_setting['strat'] == 'gt':
        result = (img > bin_setting['value'])
    elif bin_setting['strat']== 'between':
        result = ((img > bin_setting['range'][0]) * (img < bin_setting['range'][0]))
    elif bin_setting['strat']== 'not_between':
        result = ((img < bin_setting['range'][0]) + (img > bin_setting['range'][0])) > 0
    return result.float()

def get_warped_mask_set_old(bundle, res, keys_to_apply):
    result = torch.ones((1, bundle['src'].shape[-2], bundle['src'].shape[-1]),
                        device=bundle['src'].device)
    res = res.squeeze()
    for settings in keys_to_apply:
        name = settings['name']
        mask = bundle[name].squeeze()
        if 'fm' in settings:
            mask = mask[settings['fm']]
        mask = mask.unsqueeze(0)
        if 'binarization' in settings:
            mask = binarize(mask, settings['binarization'])
        mask_warp_threshold = 0.8
        if (res != 0).sum() > 0:
            mask = res_warp_img(mask.float(), res, is_pix_res=True)
            mask = (mask > mask_warp_threshold).to(dtype=torch.float32, device=result.device)

        if 'mask_value' in settings:
            result[mask != 1.0] = settings['mask_value']
        else:
            result[mask != 1.0] = 0.0

        if 'coarsen_ranges' in settings:
            around_the_mask = mask.clone()
            for length, weight in settings['coarsen_ranges']:
                around_the_mask = coarsen_mask(around_the_mask, length)
                result[(around_the_mask == 0) * (result == 1)] = weight
    return result


def get_warped_mask_set(bundle, res, keys_to_apply):
    prewarp_result = torch.ones((1, bundle['src'].shape[-2], bundle['src'].shape[-1]),
                        device=bundle['src'].device)
    res = res.squeeze()

    for settings in keys_to_apply:
        name = settings['name']
        mask = bundle[name].squeeze()
        if 'fm' in settings:
            mask = mask[settings['fm']]
        mask = mask.unsqueeze(0)
        if 'binarization' in settings:
            mask = binarize(mask, settings['binarization'])
        if 'mask_value' in settings:
            prewarp_result[mask != 1.0] = settings['mask_value']
        else:
            prewarp_result[mask != 1.0] = 0.0

        around_the_mask = mask
        if 'coarsen_ranges' in settings:
            for length, weight in settings['coarsen_ranges']:
                if length > 10:
                    around_the_mask = coarsen_mask_pool(around_the_mask, length)
                else:
                    around_the_mask = coarsen_mask(around_the_mask, length)
                prewarp_result[(around_the_mask != 1.0) * (prewarp_result == 1)] = weight

    if (res != 0).sum() > 0:
        result = res_warp_img(prewarp_result.float(), res, is_pix_res=True)
    else:
        result = prewarp_result
    return result

def get_mse_and_smoothness_masks3(bundle, **kwargs):
    mse_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0)]}
        ],
        'tgt':[
            {"name": "tgt_defects",
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0)]}
        ]
    }
    if True:
        mse_keys_to_apply['src'] += [
            {'name': 'src_large_defects',
                 'binarization': {'strat': 'value', 'value': 0},
                 "coarsen_ranges": [(1, 0)]},
                {'name': 'src_small_defects',
                 'binarization': {'strat': 'value', 'value': 0},
                 "coarsen_ranges": [(1, 0)]},
                {'name': 'src',
                 'fm': 0,
                 'binarization': {'strat': 'gt', 'value': -5.0}}
        ]
        mse_keys_to_apply['tgt'] += [
            {'name': 'tgt',
                 'fm': 0,
                 'binarization': {'strat': 'gt', 'value': -5.0}
                 }
        ]

    sm_keys_to_apply = {
        'src': [
            {'name': 'src_large_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0), (3, 2), (30, 0.3)],
             "mask_value": 0e-5},
            {'name': 'src_small_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0), (2, 2), (4, 0.3)],
             "mask_value": 0e-5},
            {'name': 'src',
             'fm': 0,
             'binarization': {'strat': 'gt', 'value': -5.0},
             'mask_value': 0
            }
            ],
        'tgt':[
        ]
    }
    if True:
        sm_keys_to_apply['src'] += [
            {'name': 'src_defects',
                 'binarization': {'strat': 'value', 'value': 0},
                 "coarsen_ranges": [(1, 0), (3, 2), (7, 0.3)],
                 "mask_value": 0e-5}
        ]
    #mse_keys_to_apply['src'] = []
    #mse_keys_to_apply['tgt'] = []
    #sm_keys_to_apply['src'] = []
    #sm_keys_to_apply['tgt'] = []
    return get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply)


def get_mse_and_smoothness_masks2(bundle, mse_keys_to_apply,
                                  sm_keys_to_apply, **kwargs):
    #yoman

    #mse_keys_to_apply['src'] = []
    #mse_keys_to_apply['tgt'] = []
    #sm_keys_to_apply['src'] = []
    #sm_keys_to_apply['tgt'] = []
    return get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply)


def get_mse_and_smoothness_masks2_poolmask(bundle, **kwargs):
    mse_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(4, 0)]}
            ],
        'tgt':[
            {'name': 'tgt_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(4, 0)]}
        ]
    }
    sm_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(4, 0)],
             "mask_value": 0e-6},
            {'name': 'src_large_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(8, 2), (64, 0.3)],
             "mask_value": 1e-6},
            {'name': 'src_small_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(8, 0.6)],
             "mask_value": 0e-6},
            {'name': 'src',
                'fm': 0,
             'binarization': {'strat': 'gt', 'value': -5.0},
            'mask_value': 0}
            ],
        'tgt':[
        ]
    }
    #mse_keys_to_apply['src'] = []
    #mse_keys_to_apply['tgt'] = []
    #sm_keys_to_apply['src'] = []
    #sm_keys_to_apply['tgt'] = []
    return get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply)

def get_mse_and_smoothness_masks(bundle,
                                 white_threshold,
                                 coarsen_mse=1, coarsen_smooth=5, coarsen_positive_mse=0,
                                 only_near_fold_mse=False, brightness_mse_range=None,
                                 tgt_defects_mse=True, tgt_defects_sm=False, positive_mse_mult=2.5,
                                 sm_mask_factor=0, sm_decay_factor=0.5, sm_decay_length=0,
                                 sm_coarsen_settings={}):
    # THIS IS TAILORED FOR NCC
    src = bundle['src']
    tgt = bundle['tgt']
    pred_res = bundle['pred_res']
    pred_tgt = bundle['pred_tgt']
    src_edges = (1 - bundle['src_edges'])
    #src_defects = ((1 - bundle['src_defects']) > 0.15).type(torch.cuda.FloatTensor)#* get_defect_mask(src_var, threshold=white_threshold)
    #tgt_defects = ((1 - bundle['tgt_defects']) > 0.15).type(torch.cuda.FloatTensor)
    binarized_src_defects = (bundle['src_defects'] > 0.01).type(torch.cuda.FloatTensor)#* get_defect_mask(src_var, threshold=white_threshold)
    src_defects = (1 - binarized_src_defects)
    tgt_defects = (1 - bundle['tgt_defects']).type(torch.cuda.FloatTensor)
    src_plastic = (1 - bundle['src_plastic'])
    tgt_plastic = (1 - bundle['tgt_plastic'])

    src_white = (src != float("inf")).float()#get_white_mask(src, threshold=white_threshold)
    tgt_white = (tgt != float("inf")).float()#get_white_mask(tgt, threshold=white_threshold)
    warped_src_edges = res_warp_img(src_edges, pred_res, is_pix_res=True)
    warped_src_plastic = res_warp_img(src_plastic, pred_res, is_pix_res=True)

    warped_src_white = (pred_tgt != float("inf")).float()#get_white_mask(pred_tgt)

    is_tissue_mask = tgt_white * warped_src_white * tgt_plastic * warped_src_plastic

    mask_warp_threshold = 0.8
    src_defects_mse = src_defects.clone()
    if coarsen_mse > 0:
        src_defects_mse = coarsen_mask(src_defects_mse, n=coarsen_mse)
    warped_src_defects_mse = res_warp_img(src_defects_mse, pred_res, is_pix_res=True)
    warped_src_defects_mse = ((warped_src_defects_mse) > mask_warp_threshold).type(torch.cuda.FloatTensor)
    defect_mask_mse = warped_src_defects_mse * tgt_defects

    mse_mask = (defect_mask_mse * warped_src_edges * is_tissue_mask)

    around_the_fold = src_defects_mse.clone()
    if coarsen_positive_mse > 0:
        around_the_fold = coarsen_mask(around_the_fold, n=coarsen_positive_mse)
    warped_src_defects_posmse = res_warp_img(around_the_fold, pred_res, is_pix_res=True)
    warped_src_defects_posmse = ((warped_src_defects_posmse) > mask_warp_threshold).type(torch.cuda.FloatTensor)
    mse_mask[(mse_mask > 0) * (warped_src_defects_posmse == 0)] = positive_mse_mult

    if only_near_fold_mse:
        mse_mask[((mse_mask > 0) * (warped_src_defects_posmse < 0.01)) == 0] = 0.0

    src_defects_smoo = (src_defects).clone()
    if coarsen_smooth > 0:
        src_defects_smoo= coarsen_mask(src_defects_smoo, n=coarsen_smooth)
    warped_src_defects_smoo = res_warp_img(src_defects_smoo, pred_res, is_pix_res=True)
    warped_src_defects_smoo = ((warped_src_defects_smoo) > mask_warp_threshold).type(
            torch.cuda.FloatTensor)
    defect_mask_smoo = warped_src_defects_smoo * tgt_defects
    smoothness_mask = defect_mask_smoo * warped_src_edges * is_tissue_mask
    smoothness_mask[smoothness_mask == 0] = sm_mask_factor

    around_the_fold = src_defects_smoo.clone()
    if sm_decay_length > 0:
        around_the_fold = coarsen_mask(around_the_fold, n=sm_decay_length)
    warped_src_defects_smoo_decay = res_warp_img(around_the_fold, pred_res, is_pix_res=True)
    warped_src_defects_smoo_decay = ((warped_src_defects_smoo_decay) > mask_warp_threshold).type(
            torch.cuda.FloatTensor)
    smoothness_mask[(smoothness_mask != sm_mask_factor) * (warped_src_defects_smoo_decay == 0)] = sm_decay_factor


    if brightness_mse_range is not None:
        tgt_brightness_mask = get_brightness_mask(tgt, low_cutoff=brightness_mse_range[0], high_cutoff=brightness_mse_range[1])
        pred_tgt_brightness_mask = get_brightness_mask(pred_tgt, low_cutoff=brightness_mse_range[0], high_cutoff=brightness_mse_range[1])
        mse_mask = mse_mask * tgt_brightness_mask * pred_tgt_brightness_mask
    #masked_out_mse_multiplier = torch.sum(torch.ones_like(mse_mask)) / torch.sum(mse_mask)
    #mse_mask *= masked_out_mse_multiplier
    #masked_out_mse_multiplier = torch.sum(torch.ones_like(mse_mask)) / torch.sum(mse_mask)
    return mse_mask, smoothness_mask

def generate_plastic_mask(img_path, out_mask_path, dataset_mip=6):
    # Plastic
    fms = 24
    arch_desc = {'fms': [1, fms, fms, fms, fms, fms, 1] , 'tags': ["plastic", "mult1", "mip8", "try2"],
                 'initc_mult': 1.0E-4 * np.sqrt(60)}

    m = create_masker(arch_desc)

    mask_mip    = 8

    data_h5 = h5py.File(img_path, 'r')
    data_dset = data_h5['main']
    dataset_dim = list(data_dset.shape)
    dataset_dim[-1] /= 2**(mask_mip - dataset_mip)
    dataset_dim[-2] /= 2**(mask_mip - dataset_mip)
    print (dataset_dim)

    downsampler = torch.nn.AvgPool2d(2, count_include_pad=False)
    upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    #plastic_h5.close()
    plastic_h5 = h5py.File(out_mask_path, 'w')
    if 'main' in plastic_h5:
        plascic_dset = plastic_h5['main']
    else:
        plascic_dset = plastic_h5.create_dataset("main", dataset_dim)

    for i in range(0, dataset_dim[0]):
        for j in range(0, dataset_dim[1]):
            s = time.time()
            #sample = np.flip(np.rot90(defect_gt_dset[i], k=1), axis=0)
            img = data_dset[i, j]

            img_var = torch.cuda.FloatTensor(img) / 255. - 0.5
            img_var = img_var.unsqueeze(0).unsqueeze(0)

            for _ in range(dataset_mip, mask_mip):
                img_var = downsampler(img_var)

            img_var_run = downsampler(downsampler(img_var))
            pred_mask_downs = m(img_var_run)
            pred_mask = upsampler(upsampler(pred_mask_downs))

            pred_mask_np = get_np(pred_mask).squeeze() > 0.9
            filtered_np = filter_connected_component(pred_mask_np, 50)
            plascic_dset[i, j] = filtered_np
            print (i, j, time.time() - s)

    plastic_h5.close()

def generate_defect_mask(img_path, out_mask_path, dataset_mip=6):
    ## Fold
    fms = 24
    arch_desc = {'fms': [1, fms, fms, fms, fms, fms, 1] , 'tags': ["fold", "mult35", "mip8"],
                 'initc_mult': 1.0E-4 * np.sqrt(60)}
    m_fold = create_masker(arch_desc)

    ## Crack
    fms = 24
    arch_desc = {'fms': [1, fms, fms, fms, fms, fms, 1] , 'tags': ["crack", "mult80", "mip8"],
                 'initc_mult': 1.0E-4 * np.sqrt(60)}

    m_crack = create_masker(arch_desc)

    mask_mip    = 8

    data_h5 = h5py.File(img_path, 'r')
    data_dset = data_h5['main']
    dataset_dim = list(data_dset.shape)
    dataset_dim[-1] /= 2**(mask_mip - dataset_mip)
    dataset_dim[-2] /= 2**(mask_mip - dataset_mip)
    print ('Dataset dim: {}'.format(dataset_dim))

    downsampler = torch.nn.AvgPool2d(2, count_include_pad=False)
    #defect_h5.close()
    defect_h5 = h5py.File(out_mask_path, 'w')
    if 'main' in defect_h5:
        defect_dset = defect_h5['main']
    else:
        defect_dset = defect_h5.create_dataset("main", dataset_dim)

    for i in range(0, dataset_dim[0]):
        for j in range(0, dataset_dim[1]):
            s = time.time()
            #sample = np.flip(np.rot90(defect_gt_dset[i], k=1), axis=0)
            img = data_dset[i, j]

            img_var = torch.cuda.FloatTensor(img) / 255. - 0.5
            img_var = img_var.unsqueeze(0).unsqueeze(0)

            for _ in range(dataset_mip, mask_mip):
                img_var = downsampler(img_var)

            pred_crack = m_crack(img_var)
            pred_crack_np = get_np(pred_crack).squeeze() > 0.9
            del pred_crack
            pred_fold  = m_fold(img_var)
            pred_fold_np = get_np(pred_fold).squeeze() > 0.9
            del pred_fold
            filtered_crack_np = filter_connected_component(pred_crack_np, 50)
            filtered_fold_np = filter_connected_component(pred_fold_np, 50)
            defects_np = np.logical_or(filtered_crack_np, filtered_fold_np)
            defect_dset[i, j] = defects_np
            print (i, j, time.time() - s)

    defect_h5.close()

def generate_edge_mask(img_path, out_mask_path, plastic_mask_path, dataset_mip=6, plastic_mip=6):
    mask_mip    = 9

    print ("Started")
    data_h5 = h5py.File(img_path, 'r')
    data_dset = data_h5['main']
    dataset_dim = list(data_dset.shape)
    dataset_dim[-1] /= 2**(mask_mip - dataset_mip)
    dataset_dim[-2] /= 2**(mask_mip - dataset_mip)
    print (dataset_dim)
    if plastic_mip is not None:
        plastic_h5 = h5py.File(plastic_mask_path, 'r')
        plastic_dset = plastic_h5['main']

    downsampler = torch.nn.AvgPool2d(2, count_include_pad=False)

    edge_h5 = h5py.File(out_mask_path, 'w')
    if 'main' in edge_h5:
        edge_dset = edge_h5['main']
    else:
        edge_dset = edge_h5.create_dataset("main", dataset_dim)

    for i in range(0, dataset_dim[0]):
        for j in range(0, dataset_dim[1]):
            #sample = np.flip(np.rot90(defect_gt_dset[i], k=1), axis=0)
            s = time.time()
            img = data_dset[i, j]
            if plastic_mip is not None:
                plastic_mask = plastic_dset[i, j]
                plastic_mask_var = torch.FloatTensor(plastic_mask).unsqueeze(0).unsqueeze(0)
                for _ in range(plastic_mip, mask_mip):
                    plastic_mask_var = downsampler(plastic_mask_var)

            img_var = torch.FloatTensor(img)
            img_var = img_var.unsqueeze(0).unsqueeze(0)

            for _ in range(dataset_mip, mask_mip):
                img_var = downsampler(img_var)

            if plastic_mip is not None:
                img_var = (img_var * (1 - plastic_mask_var)) / 255. - 0.5
            else:
                img_var = (img_var) / 255. - 0.5

            img_np = get_np(img_var)
            mask2 = find_image_edge(img_np[0, 0])
            edge_dset[i, j] = mask2
            print (i, j, time.time() - s, np.mean(mask2), np.mean(img_np))

    edge_h5.close()
    data_h5.close()

def downsample_mask(mask_in, N=5):
    dmask = skimage.measure.block_reduce(mask_in, (2,2), np.mean)
    dmask = dmask > 0.0
    fdmask = filter_connected_component(dmask, N)
    return fdmask

def downsample_image(img):
   dimg = skimage.measure.block_reduce(img, (2,2), np.mean)
   return dimg

def get_tissue_mask(img):
   mask = img < -0.05
   mask2 = filter_biggest_component(mask, only_positive=False)
   return mask2
