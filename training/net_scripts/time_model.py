#!/usr/bin/env python3
import sys
import h5py
import torch
import torchvision
import argparse
from pathlib import Path
from PIL import Image
from utilities.archive import ModelArchive
from utilities.helpers import save_chunk


def main(model_name, dset_path, dset_key, sample_number, n):
    # load model
    a = ModelArchive(model_name)
    m = a.model
    p = a.preprocessor

    # load sample
    dset = h5py.File(dset_path, 'r')
    if len(sample_number) > 1:
        sample = (dset['dense_folds_train_mip2']
                  [sample_number[0], sample_number[1]:sample_number[1] + 2])
    else:
        sample = (dset['dense_folds_train_mip2']
                  [sample_number[0], 0:2])
    sample = torch.Tensor(sample[None, ...]).cuda()

    # create directory
    dir = Path('test_model_outputs')
    dir.mkdir(exist_ok=True)

    # run model
    out = m(sample[:, 0], sample[:, 1])
    save_chunk(out[..., 0], str(dir/'x'), norm=False)
    save_chunk(out[..., 1], str(dir/'y'), norm=False)

    # run model with preprocessor
    sample_prep = p(sample).cuda()
    out_prep = m(sample_prep[:, 0], sample_prep[:, 1])
    save_chunk(out_prep[..., 0], str(dir/'prep_x'), norm=False)
    save_chunk(out_prep[..., 1], str(dir/'prep_y'), norm=False)

    # check that results haven't changed
    all_good = check_output(dir)
    if not all_good:
        sys.exit()

    print('\nRunning timed tests...')
    # run just the net
    old_t = 17.37517213821411
    t = time_net(m, sample, n)
    print('Improvement: {:.2f}%'.format((old_t - t)/old_t*100))
    # run with preprocessor
    old_t = 19.52204394340515
    t = time_net_and_preprocessor(m, p, sample, n)
    print('Improvement: {:.2f}%'.format((old_t - t)/old_t*100))


def time_function(f, name=None, on=True):
    """
    Simple decorator used for timing functions.
    More capable timing suites exist, but this suffices for many purposes.

    Can be disabled by setting `on` to False.

    Usage:
        >>> @time_function
        >>> def func(x):
        >>>     pass
    """
    if not on:
        return f
    import time
    if name is None:
        name = f.__qualname__

    def f_timed(*args, **kwargs):
        start = time.time()
        f(*args, **kwargs)
        time_diff = time.time() - start
        print('{}: {} sec'.format(name, time_diff))
        return time_diff
    return f_timed


def check_output(dir):
    """Checks whether the most recent output matches the existing truth.
    Returns true if everything checks out or if no truth was found.
    """
    truth_dir = dir / 'truth'
    if truth_dir.is_dir():
        open_png = torchvision.transforms.Compose([
            Image.open,
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
        diff_dir = dir / 'diff'
        diff_dir.mkdir(exist_ok=True)
        problems = {}
        # compare to truth outputs  # TODO: not implemented yet
        for file_name in ['x.png', 'y.png', 'prep_x.png', 'prep_y.png']:
            current = open_png(dir / file_name)
            truth = open_png(truth_dir / file_name)
            if not (current == truth).all():
                diff = (current - truth)**2
                diff_max = diff.max().item()
                problems[file_name] = diff_max
                if diff_max < 1:
                    diff /= diff_max
                save_chunk(diff, str(diff_dir/file_name.split('.')[0]))
            else:
                if (diff_dir/file_name).exists():
                    (diff_dir/file_name).unlink()
        if problems:
            print('Problems in:', problems)
            return False
        else:
            print('No problems found. Looks good!')
            return True
    else:
        print('WARNING: No truth found. Skipping.')
        return True


@time_function
def time_net(net, sample, n=1000):
    print('Running just the net {}x:'.format(n))
    for i in range(n):
        print('{}/{}'.format(i, n), end='\r')
        net(sample[:, 0], sample[:, 1])
    print('{}/{}'.format(n, n))


@time_function
def time_net_and_preprocessor(net, prep, sample, n=1000):
    print('Running the net and preprocessor {}x:'.format(n))
    for i in range(n):
        print('{}/{}'.format(i, n), end='\r')
        sample_prep = prep(sample).cuda()
        net(sample_prep[:, 0], sample_prep[:, 1])
    print('{}/{}'.format(n, n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate defects in a dset.')
    parser.add_argument(
        '--model_name', type=str,
        default='vector_fixer30',
        help='Name of the model to run')
    parser.add_argument(
        '--dset_path', type=str,
        default='/usr/people/bnehoran/training_data/mip2_mixed.h5',
        help='Path the H5 file to use')
    parser.add_argument(
        '--dset_key', type=str,
        default='dense_folds_train_mip2',
        help='key to the dataset to read from the H5 file')
    parser.add_argument(
        '--sample_number', type=str,
        default='0,0',
        help='The idex of the sample to use')
    parser.add_argument(
        '-n', type=int, default=100,
        help='Number of times to loop when timing')
    args = parser.parse_args()
    args.sample_number = [int(s) for s in args.sample_number.split(',')]

    main(args.model_name, args.dset_path, args.dset_key,
         args.sample_number, args.n)
