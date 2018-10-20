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


def parse_args(args=None):
    parser = argparse.ArgumentParser(prog='train',
                                     description='SEAMLeSS training')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')

    parallel_group = parser.add_argument_group('parallelization')
    parallel_group.add_argument(
        '--num_workers', type=int, default=1, metavar='W',
        help='Number of workers for the DataLoader',
    )
    parallel_group.add_argument(
        '--gpu_ids', type=str, default=['0'], nargs='+', metavar='X',
        help='Specific GPUs to use during training',
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

    param_group = start_parser.add_argument_group('training parameters')
    param_group.add_argument(
        '--lr', '--learning_rate', default=0.1, type=float,
        metavar='LR', help='initial learning rate',
    )
    param_group.add_argument(
        '--gamma', '--learning_rate_deccay', default=0.1, type=float,
        metavar='DR', help='rate by which the learning rate deccays',
    )
    param_group.add_argument(
        '--gamma_step', default=30, type=int, metavar='DC',
        help='frequency with which the learning rate deccay occurs',
    )
    param_group.add_argument(
        '--num_epochs', default=None, type=int, metavar='N',
        help='number of total epochs to run',
    )
    default_low, default_high = 2, 9
    param_group.add_argument(
        '--lm', '--low_mip', metavar='L',
        help='mip of lowest aligner to train', type=int,
        default=default_low,
    ).completer = (lambda **kwargs: str(default_low))
    param_group.add_argument(
        '--hm', '--high_mip', metavar='H',
        help='mip of highest aligner to train', type=int,
        default=default_high,
    ).completer = (lambda **kwargs: str(default_high))
    # param_group.add_argument(
    #     '--momentum', default=0.9, type=float, metavar='M',
    #     help='momentum')
    param_group.add_argument(
        '--wd', '--weight_decay', default=1e-4, type=float,
        metavar='W', help='weight decay (default: 1e-4)')
    # param_group.add_argument(
    #     '-A', '--skip_aug',
    #     help='skip data augmentation (no cutouts, etc)',
    #     action='store_true',
    # )  # not implemented yet. Uncomment once implemented
    # param_group.add_argument(
    #     '--plan', type=str,
    #     help='path to a training plan',
    # )  # not implemented yet. Uncomment once implemented
    param_group.add_argument(
        '--seed', default=None, type=int,
        help='seed for initializing training. '
        'Triggers deterministic behavior if specified.',
    ).completer = (lambda **kwargs: [str(random.getrandbits(10))])
    param_group.add_argument(
        '--batch_size', type=int, default=1, metavar='SIZE',
        help='Number of samples to be evaluated before each gradient update',
    )

    loss_group = start_parser.add_argument_group('training loss')
    loss_type = loss_group.add_mutually_exclusive_group()
    loss_type.add_argument(
        '-s', '--supervisied',
        help='Train in a supervised fashion on randomly generated ground '
             'truth vector fields and false slice pairs. This is the default.',
        dest='supervised',
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
        type=str, default='jacob',
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
        type=str, default=None,
    )

    checkpoint_group = start_parser.add_argument_group('checkpointing')
    checkpoint_group.add_argument(
        '--log', '--log_interval', '--log_time',
        help='the number of samples in between each log. '
             'Use 0 to disable.',
        dest='log_time',
        type=int, default=10, metavar='T',
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
        type=int, default=100, metavar='T',
    )
    checkpoint_group.add_argument(
        '-i', '--interval',
        help='combines the log, checkpoint, and visualization intervals',
        dest='interval',
        type=int, default=None, metavar='T',
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args)
    if 'interval' in args and args.interval is not None:
        args.log_time = args.interval
        args.checkpoint_time = args.interval
        args.vis_time = args.interval
        del args.interval
    return args


def _list_trained_nets(**kwargs):
    """
    Returns a list of available trained nets.
    """
    return next(os.walk('../models/'))[1]


if __name__ == '__main__':
    print(parse_args())
