#!/usr/bin/env python3
"""
T Macrina
181204

Make 3D stack of TIF images in a directory (sorted), crop, & save as H5

Args:
    sys.argv[1]: full path to the TIF image directory
    sys.argv[2]: filename of the H5 file (placed in the TIF directory)

Returns:
    H5 file saved in the same directory, in z,y,x order
"""

from PIL import Image
import numpy as np
import h5py
import os
import sys


def tif_to_array(fn):
    """Open TIF image and convert to numpy ndarray of dtype

    Currently tested for only for uint8 -> uint8, uint32 or uint24 -> uint32

    Args:
        fn: filename (full path) of the image

    Returns:
        An ndarray of dtype
    """
    img = np.array(Image.open(fn))
    if len(img.shape) == 3:
        img = np.dstack((np.zeros(img.shape[:2]+(1,)), img))
        img = img[:,:, ::-1]
        img = img.astype(np.uint8).view(np.uint32)
        img = img.reshape(img.shape[:2])
    return img


def make_array_from_tif_dir(dir):
    """Combine all TIF images in directory (sorted) into 3D ndarray

    Args:
        dir: folder path

    Returns:
        An ndarray of dtype with at least 3 dimesnions
    """
    array = None
    files = os.listdir(dir)
    files.sort()
    for i, file in enumerate(files):
        if file.endswith(".tif") or file.endswith(".png"):
            a = tif_to_array(os.path.join(dir, file))
            a = a[:1024,:1024]
            if array is None:
                array = a
            else:  # TODO: currently runs in quadratic time because of concat
                array = np.dstack((array, a))
            print(i, end='\r')
    return array


def write_to_h5(fn, arr):
    """Write ndarray to H5 file under group "main"
    """
    with h5py.File(fn, "w") as f:
        f.create_dataset("/main", data=arr, dtype=arr.dtype)


def main(dirs, fn):
    """Make HDF5 3D matrix of TIF images in a sorted directory & cropped
    """
    cat_array = None
    for dir in dirs:
        print('Processing {}'.format(dir))
        array = make_array_from_tif_dir(dir)
        array = array.transpose((2,1,0))
        array = array[:, np.newaxis, ...]
        if cat_array is None:
            cat_array = array
        else:  # TODO: currently runs in quadratic time because of concat
            cat_array = np.concatenate((cat_array, array), 1)
    write_to_h5(fn, cat_array)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        dir = sys.argv[1]  # seungmount/Omni/TracerTasks/minnie/masks/mip2/{}19714
        dirs = [dir.format(x) for x in ('image', 'cracks ', 'folds ')]
        fn = sys.argv[2]
        main(dirs, fn)
    else:
        print('Include input directory & output filename')
