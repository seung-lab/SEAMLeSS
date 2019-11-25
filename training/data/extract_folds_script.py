#!/usr/bin/env python3
# simple script to narrow down the dataset to only include folds
import h5py
import numpy as np
import matplotlib.pyplot as plt  # noqa: 402
plt.switch_backend('agg')
dataset_in = './100slice_minnie_1536_mip4_annotated.h5'
dataset_out = './filtered.h5'
f = h5py.File(dataset_in, 'r')
dset = f['main']
folds = dset[:, 4:]
ind = np.arange(100)[(((folds > .7).sum((1, 2, 3)) / 2359296) > .0022)]
for i in ind:
    pair = dset[i]
    for j, image in enumerate(pair):
        if j in [0, 1, 4, 5]:
            plt.imsave('temp/fold{}_{}.png'.format(i, j),
                       1-image, cmap='Greys')
# further narrow down by hand if desired:
# ind = [0,  2,  4, 15, 16, 17, 24, 25, 30,
#        39, 40, 45, 48, 56, 68, 70, 75, 89, 99]
only_folds = dset[ind, ...]
with h5py.File(dataset_out, 'w') as g:
    g.create_dataset('main', data=only_folds)
# for i, pair in enumerate(only_folds):
#     for j, image in enumerate(pair):
#         if j in [0, 1, 4, 5]:
#             plt.imsave('temp/fold_narrowed{}_{}.png'.format(i,j),
#                        1-image, cmap='Greys')
