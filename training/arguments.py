# PYTHON_ARGCOMPLETE_OK
"""Provide a command line API for supervised and unsupervised training

Defines and handles command line arguments and tab completion.
"""

import argparse
import argcomplete
import os
import random

net_name_prefixes = [
    'seamless_',
]


def _parse_args(args=None):
    parser = argparse.ArgumentParser(prog='train',
                                     description='SEAMLeSS training')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')

    parallel_group = parser.add_argument_group('parallelization')
    parallel_group.add_argument(
        '--num_workers', type=int, default=None, metavar='W',
        help='Number of workers for the DataLoader',
    )
    parallel_group.add_argument(
        '--gpu_ids', type=str, default=None, metavar='X',
        help='GPUs to use during training, separated by commas. '
             'If not specified, the first unused GPU will be used.',
    )
    parallel_group.add_argument(
        '--batch_size', type=int, default=None, metavar='SIZE',
        help='Number of samples to be evaluated before each gradient update',
    )

    resume_help = ('Resume training a paused model '
                   'using the saved training parameters.')
    resume_parser = subparsers.add_parser('resume', help=resume_help,
                                          description=resume_help)
    resume_parser.add_argument(
        'name',
        help='the saved model to resume training',
        type=str,
        metavar='MODEL',
    ).completer = _list_trained_nets

    start_help = ('Start a new training run to train a net from scratch or '
                  'from a saved state.')
    start_parser = subparsers.add_parser('start', help=start_help,
                                         description=start_help)
    start_parser.add_argument(
        'name', type=str, metavar='model_name',
        help='the name for the new model',
    ).completer = (lambda **kwargs: net_name_prefixes)
    start_parser.add_argument(
        '--saved_model',
        help='saved model to initialize with',
        type=str, default=None, metavar='MODEL',
    ).completer = _list_trained_nets
    start_parser.add_argument(
        '--height',
        help='the number of mip levels to train',
        type=int, default=5, metavar='H',
    )
    start_parser.add_argument(
        '--feature_maps', '--feature_list', '--fm',
        help='the number of feature maps at each mip level',
        type=int, nargs='+', default=[], metavar='F',
    )
    start_parser.add_argument(
        '--encodings',
        help='whether to use encodings or plain images',
        default=False,
        action='store_true',
    )

    param_group = start_parser.add_argument_group('training parameters')
    param_group.add_argument(
        '--lr', '--learning_rate', default=0.000002, type=float,
        metavar='LR', help='initial learning rate',
    )
    param_group.add_argument(
        '--gamma', '--learning_rate_decay', default=1, type=float,
        metavar='DR', help='rate by which the learning rate decays',
    )
    param_group.add_argument(
        '--gamma_step', default=30, type=int, metavar='DC',
        help='frequency with which the learning rate decay occurs',
    )
    param_group.add_argument(
        '--num_epochs', default=1000000, type=int, metavar='N',
        help='number of total epochs to run',
    )
    param_group.add_argument(
        '--plan',
        choices=('all', 'top_down', 'bottom_up', 'random_one'),
        help='determines the training order',
        type=str, default='random_one',
    )
    param_group.add_argument(
        '--epochs_per_mip', default=4, type=int, metavar='N',
        help='number of epochs to run before switching mip levels',
    )
    # default_low, default_high = 2, 9
    # param_group.add_argument(
    #     '--lm', '--low_mip', metavar='L',
    #     help='mip of lowest aligner to train', type=int,
    #     default=default_low,
    # ).completer = (lambda **kwargs: str(default_low))
    # param_group.add_argument(
    #     '--hm', '--high_mip', metavar='H',
    #     help='mip of highest aligner to train', type=int,
    #     default=default_high,
    # ).completer = (lambda **kwargs: str(default_high))
    # param_group.add_argument(
    #     '--momentum', default=0.9, type=float, metavar='M',
    #     help='momentum')
    param_group.add_argument(
        '--wd', '--weight_decay', default=0, type=float,
        metavar='W', help='weight decay (default: 0)')
    param_group.add_argument(
        '-A', '--skip_aug',
        dest='skip_aug',
        help='skip data augmentation (no cutouts, etc)',
        action='store_true',
    )  # TODO: not fully implemented yet.
    # param_group.add_argument(
    #     '--plan', type=str,
    #     help='path to a training plan',
    # )  # not implemented yet. Uncomment once implemented
    param_group.add_argument(
        '--seed', default=None, type=int,
        help='seed for initializing training. '
        'Triggers deterministic behavior if specified.',
    ).completer = (lambda **kwargs: [str(random.getrandbits(10))])

    loss_group = start_parser.add_argument_group('training loss')
    loss_type = loss_group.add_mutually_exclusive_group()
    loss_type.add_argument(
        '-s', '--supervised',
        help='Train in a supervised fashion on randomly generated ground '
             'truth vector fields and false slice pairs. This is the default.',
        dest='supervised',
        default=True,
        action='store_true',
    )
    loss_type.add_argument(
        '-u', '--unsupervised',
        help='Train on real slice pairs based on MSE and smoothness '
             'penalties.',
        dest='supervised',
        action='store_false',
    )
    loss_group.add_argument(
        '--lambda1',
        help='smoothness penalty coefficient. '
             'Only relevant for unsupervised trainng.',
        type=float, default=0.1, metavar='L1',
    )
    loss_group.add_argument(
        '--penalty',
        choices=('lap', 'jacob', 'cjacob', 'tv'),
        help='type of smoothness penalty. '
             'Only relevant for unsupervised trainng.',
        type=str, default='lap',
    )
    loss_group.add_argument(
        '--defect_net', metavar='NET',
        help='defect net archive. Indicates where to relax MSE and/or '
        'smoothness penalies in unsupervised training',
        type=str, default=None,
    )

    data_group = start_parser.add_argument_group('data')
    data_group.add_argument(
        '--validation_set', metavar='SET',
        help='Dataset to use for validation (default: None)',
        type=str, default=None,
    )
    data_group.add_argument(
        '--training_set', metavar='SET',
        help='Dataset to use for training (default: None)',
        type=str, required=True,
    )
    data_group.add_argument(
        '--num_samples', metavar='N',
        help='Number of samples from the dataset to train on. '
        'Default is all.',
        type=int, default=None,
    )

    checkpoint_group = start_parser.add_argument_group('checkpointing')
    checkpoint_group.add_argument(
        '--log', '--log_interval', '--log_time',
        help='the number of samples in between each log. '
             'Use 0 to disable.',
        dest='log_time',
        type=int, default=1, metavar='T',
    )
    checkpoint_group.add_argument(
        '--cpoint', '--checkpoint_interval', '--checkpoint_time',
        help='the number of samples in between each checkpoint. '
             'Use 0 to disable.',
        dest='checkpoint_time',
        type=int, default=100, metavar='T',
    )
    checkpoint_group.add_argument(
        '--vis', '--visualization_interval', '--vis_time',
        help='the number of samples in between each visualization. '
             'Use 0 to disable.',
        dest='vis_time',
        type=int, default=50, metavar='T',
    )
    checkpoint_group.add_argument(
        '-i', '--interval',
        help='combines the log, checkpoint, and visualization intervals, '
             'if set',
        dest='interval',
        type=int, default=None, metavar='T',
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args)

    # autocomplete some arguments
    if 'interval' in args and args.interval is not None:
        args.log_time = args.interval
        args.checkpoint_time = args.interval
        args.vis_time = args.interval
        del args.interval
    if args.gpu_ids is None:
        args.gpu_ids = first_unused_gpu()
    if args.num_workers is None:
        args.num_workers = len(args.gpu_ids.split(','))
    if 'batch_size' in args and args.batch_size is None:
        args.batch_size = len(args.gpu_ids.split(','))
    if 'test_' in args.name:  # random names for rapid testing
        while '*' in args.name:  # all '*'s are replaced by a random hex digit
            args.name = args.name.replace(
                '*', hex(random.getrandbits(4))[2:], 1)
    if 'feature_maps' in args:
        if len(args.feature_maps) == 0:
            default_num_fm = 12
            args.feature_maps = [default_num_fm] * args.height
        elif len(args.feature_maps) == 1:
            args.feature_maps = args.feature_maps * args.height
        else:
            args.height = len(args.feature_maps)
    return args


def _list_trained_nets(**kwargs):
    """
    Returns a list of available trained nets.
    """
    return next(os.walk('../models/'))[1]


def first_unused_gpu(threshold=0.05):
    """
    Returns the first unused GPU, where usage is thresholded by `threshold`.
    If none are available, returns the one with the least usage.

    Adapted from
    https://github.com/awni/cuthon
    """
    import subprocess
    try:
        nv_stats = subprocess.check_output('nvidia-smi -x -q'.split())
    except OSError:
        print('No GPUs found. Falling back to CPU.')
        return ''
    import xml.etree.ElementTree as ElementTree
    gpus = ElementTree.fromstring(nv_stats).findall('gpu')
    least = -1, 1.0
    for i, gpu in enumerate(gpus):
        mem = gpu.find('fb_memory_usage')
        tot = int(mem.find('total').text.split()[0])
        used = int(mem.find('used').text.split()[0])
        usage = used / tot
        if usage < threshold:  # this gpu is unused, so return it
            print('Using GPU {}'.format(i))
            return str(i)
        if usage < least[1]:
            least = i, usage
    # no available GPUs, so return the one with the least usage
    if least[0] >= 0:
        print('Using GPU {}'.format(least[0]))
        return str(least[0])


# parse the arguments during import to eliminate delay
_args = None
try:
    _args = _parse_args()
except Exception:
    _args = None


def parse_args(args=None):
    global _args
    if args is not None:
        _args = _parse_args(args)
    return _args


if __name__ == '__main__':
    print(_args)
