import cloudvolume as cv
import scipy
import json
import torch
import numpy as np
import time
import artificery
import copy
import os
import sys
import scipy.ndimage
import pathlib

from pdb import set_trace as st
from fcorr import get_fft_power2, get_hp_fcorr
from blockmatch import block_match

def get_np(pt):
    return pt.cpu().detach().numpy()

def normalize(img,
              per_feature_center=True,
              per_feature_var=False,
              eps=1e-5,
              mask=None,
              mask_fill=None,
            ):
    img_out = img.clone()
    if mask is not None:
        assert mask.shape == img.shape
    for i in range(1):
        for b in range(img.shape[0]):
            x = img_out[b]
            if per_feature_center and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        x[f][m] = x[f][m].clone() - torch.mean(x[f][m].clone())
                    else:
                        x[f] = x[f].clone() - torch.mean(x[f].clone())
            else:
                if mask is not None:
                    m = mask[b]
                    x[m] = x[m].clone() - torch.mean(x[m].clone())
                else:
                    x[...] = x.clone() - torch.mean(x.clone())

            if per_feature_var and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        var = torch.var(x[f][m].clone())
                        x[f][m] = x[f][m].clone() / (torch.sqrt(var) + eps)
                    else:
                        var = torch.var(x[f].clone())
                        x[f] = x[f].clone() / (torch.sqrt(var) + eps)
            else:
                if mask is not None:
                    m = mask[b]
                    var = torch.var(x[m].clone())
                    x[m] = x[m].clone() / (torch.sqrt(var) + eps)
                else:
                    var = torch.var(x.clone())
                    x[...] = x.clone() / (torch.sqrt(var) + eps)

    if mask is not None and mask_fill is not None:
        img_out[mask == False] = mask_fill

    return img_out


def create_model(name, checkpoint_folder):
    a = artificery.Artificery()

    spec_path = os.path.join(checkpoint_folder, "model_spec.json")
    my_p = a.parse(spec_path)

    checkpoint_path = os.path.join(checkpoint_folder, "{}.state.pth.tar".format(name))
    if os.path.isfile(checkpoint_path):
        my_p.load_state_dict(torch.load(checkpoint_path))
    my_p.name = name
    return my_p.cuda()

def rechunck_image(chunk_size, image):
    I = image.split(chunk_size, dim=2)
    I = torch.cat(I, dim=0)
    I = I.split(chunk_size, dim=3)
    return torch.cat(I, dim=1)

def misalignment_detector(img1, img2, mip, np_out=True, threshold=None):
    '''
        img1, img2 -- pytorch tensors, 2D (only X and Y)
        mip -- integer
    '''
    TARGET_MIP = 4
    if mip > TARGET_MIP:
        raise Exception("Misalignment detector only works for images with MIP <= 4")
    if img1.max() > 10:
        img1 /= 255.0
        img2 /= 255.0
        print ("DIVIDING")


    img1_downs = img1
    img2_downs = img2
    while len(img1_downs.shape) < 4:
        img1_downs = img1_downs.unsqueeze(0)
        img2_downs = img2_downs.unsqueeze(0)

    for _ in range(mip, TARGET_MIP):
        img1_downs = torch.nn.functional.avg_pool2d(img1_downs, 2)
        img2_downs = torch.nn.functional.avg_pool2d(img2_downs, 2)
    img1_downs_norm = normalize(img1_downs, mask=img1_downs.abs()>0.25, mask_fill=-20)
    img2_downs_norm = normalize(img2_downs, mask=img2_downs.abs()>0.25, mask_fill=-20)

    mypath = str(pathlib.Path(__file__).parent.absolute())
    pyramid_name = 'ncc_m4'
    ncc_model_path = os.path.join(mypath, "models/{}".format('ncc_m4'))
    encoder = create_model(
        "model", checkpoint_folder=ncc_model_path
    )


    with torch.no_grad():
        img1_enc = encoder(img1_downs_norm).squeeze()
        img2_enc = encoder(img2_downs_norm).squeeze()
        img1_enc[img1_downs.squeeze().abs() < 0.05] = 0
        img2_enc[img2_downs.squeeze().abs() < 0.05] = 0
        #img1_enc[img1_enc.squeeze().abs() < 0.15] = 0
        #img2_enc[img2_enc.squeeze().abs() < 0.15] = 0

    misalignment_mask = compute_fcorr(img1_enc, img2_enc).squeeze()
    #misalignment_mask_ups = scipy.misc.imresize(misalignment_mask, get_np(img1).shape)
    #fcorr_ups_var = torch.Tensor(fcorr_ups, device=img1.device)
    return misalignment_mask

def compute_fcorr(image1, image2):
    while len(image1.shape) < 4:
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
    s = time.time()
    s = time.time()
    tile_size = 128 * 2
    ma_length = 6
    bm_result = block_match(image2, image1, min_overlap_px=40000, tile_step=tile_size//2,
            tile_size=tile_size,
            peak_ratio_cutoff=4.5, peak_distance=ma_length,  max_disp=48, filler=250)
    nonzero_bm_mask1 = ((bm_result[..., 0].abs() > ma_length) + (bm_result[..., 1].abs() > ma_length)) > 0
    bm_result = block_match(image1, image2, min_overlap_px=40000, tile_step=tile_size//2,
            tile_size=tile_size,
            peak_ratio_cutoff=4.0, peak_distance=ma_length,  max_disp=48, filler=250)
    nonzero_bm_mask2 = ((bm_result[..., 0].abs() > ma_length) + (bm_result[..., 1].abs() > ma_length)) > 0
    nonzero_bm_mask = (nonzero_bm_mask1 + nonzero_bm_mask2) > 0

    print (time.time() - s, 'Masked misalignments: ', nonzero_bm_mask.sum())
    return nonzero_bm_mask
    #mult = image1*image2
    #result = (mult - mult.mean()) / torch.sqrt(mult.var())
    #result += result.min()
    #return get_np(result).squeeze()

    std1 = image1[image1!=0].std()
    std2 = image2[image2!=0].std()
    scaling = 8 * pow(std1*std2, 1/2)
    fcorr_chunk_size = 8
    #print(image1)
    new_image1 = rechunck_image(fcorr_chunk_size, image1)
    new_image2 = rechunck_image(fcorr_chunk_size, image2)

    f1, p1 = get_fft_power2(new_image1)
    f2, p2 = get_fft_power2(new_image2)
    tmp_image = get_hp_fcorr(f1, p1, f2, p2, scaling=scaling)
    tmp_image = tmp_image.permute(2,3,0,1)
    tmp_image = tmp_image.cpu().numpy()
    tmp = copy.deepcopy(tmp_image)
    tmp[tmp==2]=1
    blurred = scipy.ndimage.morphology.filters.gaussian_filter(tmp, sigma=(0, 0, 1, 1))
    s = scipy.ndimage.generate_binary_structure(2, 1)[None, None, :, :]
    closed = scipy.ndimage.morphology.grey_closing(blurred, footprint=s)
    closed = 2*closed
    closed[closed>1] = 1
    closed = 1-closed
    #print("++++closed shape",closed.shape)
    #print (np.mean(closed))
    return closed

def get_np(pt):
    return pt.cpu().detach().numpy()


