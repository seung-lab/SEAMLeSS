from scipy import ndimage
from skimage import measure
import fastremap
import kimimaro
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def postprocess(img, thr_binarize=0, w_connect=0, thr_filter=0, w_dilate=0):

	if thr_binarize:
		img = threshold_image(img, abs(thr_binarize))
		if thr_binarize<0:
			img = 1 - img
	if w_connect:
		img = dilate_mask(img, w_connect)
	if thr_filter:
		img = filter_mask(img, thr_filter)
	if w_dilate:
		img = dilate_mask(img, w_dilate)
	return img


def count_fold_pixels(img):
	cc_fold = ndimage.label(img)[0].astype(np.uint32)
	uni, uni_count = np.unique(cc_fold, return_counts=True)
	fold_to_pixel_count = dict(zip(uni[1:], uni_count[1:]))
	fastremap.remap(cc_fold, fold_to_pixel_count, preserve_missing_labels=True, in_place=True)
	return cc_fold


def postprocess_length_filter(img, thr_binarize=0, w_connect=0, thr_filter=0, return_skeleys=False, count_pixels=False):

	if thr_binarize:
		img = threshold_image(img, abs(thr_binarize))
		if thr_binarize<0:
			img = 1 - img
	if w_connect:
		img = dilate_mask(img, w_connect)

	if count_pixels:
		return count_fold_pixels(img)

	return length_filter_mask(img, return_skeleys)


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
	if fold_num.shape[0] > 1 or np.sum(fold_num) == 0:
		fold_num = fold_num[1:]; fold_size = fold_size[1:]
	img_lab_vec = np.reshape(img_lab, (-1,))
	img_relab = np.reshape(fastremap.remap_from_array_kv(img_lab_vec, fold_num, fold_size), img_lab.shape)

	return (img_relab>=size_thr).astype('uint8')

def length_filter_mask(img, return_skeleys=False):
	print('skeletonizing...')
	img_lab = measure.label(img).astype(np.uint32)
	fold_num = np.unique(img_lab)
	DEFAULT_TEASAR_PARAMS = {
  		'scale': 0.5,
		'const': 1,
  		'pdrf_scale': 100000,
  		'pdrf_exponent': 4,
	}
	skels = kimimaro.skeletonize(img_lab, teasar_params=DEFAULT_TEASAR_PARAMS, dust_threshold=0)
	if return_skeleys:
		return_img = np.zeros(shape=img.shape, dtype=np.uint32)
		if skels != {}:
			for key in skels:
				len = skels[key].cable_length()
				rows = skels[key].vertices[:,0].astype(np.int)
				cols = skels[key].vertices[:,1].astype(np.int)
				return_img[rows,cols] = np.uint32(len)
		return return_img
	else:
		remap_dict = dict(zip(fold_num, np.zeros(shape=fold_num.shape, dtype=np.uint32)))
		if skels != {}:
			for key in skels:
				len = skels[key].cable_length()
				remap_dict[skels[key].id] = np.uint32(len)
		fastremap.remap(img_lab, remap_dict, in_place=True, preserve_missing_labels=True)
		return img_lab 