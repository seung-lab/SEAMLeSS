import h5py
import numpy as np
import argparse
from os.path import expanduser, join

def mask_to_slices(mask):
	"""Given 1D bit array, return slices of 2+ True
	"""
	o = []
	a, b = -1, -1
	for i in range(len(mask)-1):
		if a >= 0 and not mask[i+1]:
			b = i+1
			if b-a > 1:
				o.append(slice(a, b))
			a, b = -1, -1
		if a == -1 and mask[i] and mask[i+1]:
			a = i
	if a >= 0 and b == -1:
		o.append(slice(a, len(mask)))
	return o

def split_stack(stack, nonzero_frac=0.8, std_threshold=10):
	"""Given 3D image, produce array of 3D images of contiguous sections that
	meet cleaning criteria.
	"""
	splits = []
	sum_mask = np.sum(stack > 0, axis=(1,2))
	num_pixels = stack.shape[1]*stack.shape[2]
	std_mask = np.std(stack, axis=(1,2))
	mask = (sum_mask > num_pixels*nonzero_frac) & (std_mask > std_threshold)
	slices = mask_to_slices(mask)
	for sl in slices:
		splits.append(stack[sl])
	return splits
	

def main(src_fn, dst_fn, nonzero_frac=0.8, std_threshold=10):
	"""Clean H5 by separating dsets into stacks of nonzero images

	Args:
	* src_fn: path to H5 to be cleaned
	* dst_fn: path where cleaned H5 will be written
	* nonzero_frac: Minimum fraction of nonzero pixels that must exist in each 
					section for it to not be removed.
	* std_threshold: Minimum standard deviation that each section must have for
					for it to not be removed.
	"""
	print('Cleaning H5 {0}'.format(src_fn))
	src = h5py.File(src_fn, 'r')
	dst = h5py.File(dst_fn, 'w')
	for src_k in src.keys():
		print('Cleaning dset {0}'.format(src_k))
		n = 0
		dset = src[src_k]
		for i in range(dset.shape[0]):
			src_stack = dset[i]
			splits = split_stack(src_stack, nonzero_frac=nonzero_frac, 
											std_threshold=std_threshold)
			if len(splits) > 0:
				for split in splits:
					dst_k = '{0}_{1:03}'.format(src_k, n)
					print('Writing dset {0}, {1}'.format(dst_k, split.shape))
					dst.create_dataset(dst_k, data=split[np.newaxis, ...])
					n += 1
	dst.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Create score image.')
  parser.add_argument('--src_path', type=str, 
    help='Path to H5 to be cleaned')
  parser.add_argument('--dst_path', type=str,
    help='Path where cleaned H5 will be written')
  parser.add_argument('--nonzero_frac', type=int, default=0.8,
    help='Minimum fraction of nonzero pixels that must exist in each section.')
  parser.add_argument('--std_threshold', type=int, default=10,
    help='Minimum standard deviation that each section must have.')
  args = parser.parse_args()

  main(args.src_path, args.dst_path, args.nonzero_frac, args.std_threshold)

# src_fn = join(expanduser('~'), 'seungmount/research/eam6/data/test_mip2_drosophila.h5')
# dst_fn = join(expanduser('~'), 'seungmount/research/eam6/data/test_mip2_drosophila_cleaned.h5')