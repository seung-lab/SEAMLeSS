import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import fastremap

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def postprocess(img, thr_binarize=0, w_connect=0, thr_filter=0, w_dilate=0):

	if thr_binarize:
    	img = threshold_image(img, thr_binarize)
    if w_connect:
    	img = dilate_mask(img, w_connect)
    if :
    	img = filter_mask(img, thr_filter)
    if dilate_w:
    	img = dilate_mask(img, w_dilate)

    return img


def threshold_image(img, thr):

	# Fix threshold
	if np.max(img) > 10 and thr < 1:
		new_thr = 255*thr
	elif np.max(img) < 10 and thr > 1:
		new_thr = thr/255.0
	else:
		new_thr = thr

	return (img>=new_thr).astype('uint8')


def dilate_mask(img, w_dilate):

	if np.max(img) > 10:
		img = (img/np.max(img)).astype('uint8')

	struct = np.ones((w_dilate,w_dilate), dtype=bool)

	return ndimage.binary_dilation(img, structure=struct).astype(img.dtype)

def filter_mask(img, size_thr):

	img_lab = measure.label(img)

	fold_num, fold_size = np.unique(img_lab, return_counts=True)
	fold_num = fold_num[1:]; fold_size = fold_size[1:]

	img_lab_vec = np.reshape(img_lab, (-1,))
	img_relab = np.reshape(fastremap.remap_from_array_kv(img_lab_vec, fold_num, fold_size), img_lab.shape)

	return (img_relab>=size_thr).astype('uint8')
