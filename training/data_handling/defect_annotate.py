#!/usr/bin/env python3
"""Add defect annotations to a training dataset
Sample command:
    >>> ./defect_annotate.py --src_path ~/training_data/dataset1.h5 \
        --dst_path ~/training_data/dataset1_annotated.h5 \
        --defect_net defect_net_v01
"""
import torch
import h5py
import argparse
from utilities.archive import ModelArchive
from utilities.helpers import downsample


def run_net_chunked(net, image, chunks=(2, 2)):
    """Run a net on a given image in a chunked fashion

    Args:
    * net: net to run
    * image: image to process
    * chunks: the number of chunks to divide the image into
    """
    image = torch.Tensor(image)
    chunks = [c.chunk(chunks[1], 1) for c in image.chunk(chunks[0], 0)]
    cracks, folds = [], []
    for row in chunks:
        crack_row, fold_row = [], []
        for c in row:
            c = c.unsqueeze(0).unsqueeze(0)
            c = net.preprocessor(c)
            c_c, c_f = net.model(c.cuda()).cpu()[0]
            crack_row.append(c_c), fold_row.append(c_f)
        cracks.append(crack_row), folds.append(fold_row)
    cracks = [torch.cat(crack_row, 1) for crack_row in cracks]
    cracks = torch.cat(cracks, 0)
    folds = [torch.cat(fold_row, 1) for fold_row in folds]
    folds = torch.cat(folds, 0)
    return image, cracks, folds


def main(src_fn, dst_fn, defect_net_name='minnie_mip2_defect_v03'):
    """Add defect annotations to a training dataset

    Args:
    * src_fn: path to H5 to be processed
    * dst_fn: path where the new H5 will be written
    """
    defect_net = ModelArchive(defect_net_name)
    down = downsample(2)
    print('Processing H5 {0}'.format(src_fn))
    with h5py.File(src_fn, 'r') as src, h5py.File(dst_fn, 'w') as dst:
        for src_k in src.keys():
            print('Processing dset {0}'.format(src_k))
            n = 0
            src_dset = src[src_k]
            dst_dset = dst.create_dataset(
                src_k, (src_dset.shape[0], 6, 1536, 1536), dtype='f')
            for i in range(src_dset.shape[0]):
                a, b = src_dset[i]
                a, a_c, a_f = run_net_chunked(defect_net, a)
                b, b_c, b_f = run_net_chunked(defect_net, b)
                a = down(a.unsqueeze(0)).squeeze(0).numpy()
                b = down(b.unsqueeze(0)).squeeze(0).numpy()
                a_c = down(a_c.unsqueeze(0)).squeeze(0).numpy()
                b_c = down(b_c.unsqueeze(0)).squeeze(0).numpy()
                a_f = down(a_f.unsqueeze(0)).squeeze(0).numpy()
                b_f = down(b_f.unsqueeze(0)).squeeze(0).numpy()
                dst_dset[n] = a, b, a_c, b_c, a_f, b_f
                n += 1
                print('Progress {}/{}'.format(n, src_dset.shape[0]), end='\r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate defects in a dset.')
    parser.add_argument(
        '--src_path', type=str,
        help='Path to H5 to be processed')
    parser.add_argument(
        '--dst_path', type=str,
        help='Path where the new H5 will be written')
    parser.add_argument(
        '--defect_net', type=str,
        help='The name of the defect net to use')
    args = parser.parse_args()

    main(args.src_path, args.dst_path, args.defect_net)
