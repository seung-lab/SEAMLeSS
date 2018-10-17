#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Train a network.

This is the main module which is invoked to train a network.

Running:
    To begin training, run

        $ python3 supervised_train.py start MODEL_NAME [...]

    or equivalently

        $ ./supervised_train.py start MODEL_NAME [...]

    To get help with the command line options, use

        $ python3 supervised_train.py start --help

Resuming:

    It is possible to resume training. This is useful if a training run was
    killed by accident or circumstance and you would like to continue training.
    Before doing this, check to make sure the run is actually in fact dead,
    since attempting to resume a live run could have undefined behavior.
    To resume training run

        $ python3 supervised_train.py resume MODEL_NAME

    where `MODEL_NAME` is the name of the previously stopped training run.

Example:
        $ python3 supervised_train.py start improved_net --lm 2 --hm 9

Editor's note:

    The top of this file should contain the comment `# PYTHON_ARGCOMPLETE_OK`
    in order to allow command line tab completion to work properly.

"""

import os
import time
import warnings
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

from arguments import parse_args  # TODO: move up for faster arg access
from archive import ModelArchive, warn_change
import stack_dataset as stack
from helpers import gridsample_residual

best_prec1 = 0


def main():
    global args, state_vars, best_prec1
    args = parse_args()

    # set available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    # create or load the model, optimizer, and parameters
    if args.command == 'start':  # TODO: factor out
        if ModelArchive.model_exists(args.name):
            raise ValueError('The model "{}" already exists.'.format(args.name))
        archive = ModelArchive(args.name, readonly=False, seed=args.seed)
        model = archive.model

        start_lr = args.lr
        lr = start_lr
        wd = args.wd
        optimizer = archive.optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = wd

        start_epoch = 0
        num_epochs = args.num_epochs
        training_set_path = Path(args.training_set)
        validation_set_path = Path(args.validation_set)
        lm = args.lm
        hm = args.hm
        supervised = args.supervised
        batch_size = args.batch_size
        log_time = args.log_time
        cpoint = args.cpoint
        vis = args.vis
    else:  # args.command == 'resume'
        archive = ModelArchive(args.name, readonly=False)
        model = archive.model

        start_lr = archive.state_vars['start_lr']
        lr = archive.state_vars['lr']
        wd = archive.state_vars['wd']
        optimizer = archive.optimizer
        for param_group in optimizer.param_groups:
            warned_lr = False
            if param_group['lr'] != lr:
                if not warned_lr:
                    warn_change('learning rate', param_group['lr'], lr)
                    warned_lr = True
                param_group['lr'] = lr
            warned_wd = False
            if param_group['weight_decay'] != wd:
                if not warned_wd:
                    warn_change('weight decay', param_group['weight_decay'], wd)
                    warned_wd = True
                param_group['weight_decay'] = wd

        start_epoch = archive.state_vars['epoch']
        num_epochs = archive.state_vars['num_epochs']
        training_set_path = Path(archive.state_vars['training_set'])
        validation_set_path = Path(archive.state_vars['validation_set'])
        lm = archive.state_vars['lm']
        hm = archive.state_vars['hm']
        supervised = archive.state_vars['supervised']
        batch_size = archive.state_vars['batch_size']
        log_time = archive.state_vars['log_time']
        cpoint = archive.state_vars['cpoint']
        vis = archive.state_vars['vis']

    # update the archive
    state_vars = {
        'name': args.name,
        'epoch': start_epoch,
        'num_epochs': num_epochs,
        'training_set': str(training_set_path),
        'validation_set': str(validation_set_path),
        'start_lr': start_lr,
        'lr': lr,
        'weight_decay': wd,
        'lm': lm,
        'hm': hm,
        'supervised': supervised,
        'batch_size': batch_size,
        'log_time': log_time,
        'cpoint': cpoint,
        'vis': vis,
    }
    archive.update(**state_vars)
    log_titles = [
        'Time Stamp',
        'Epoch',
        'Iteration',
        'Training Loss',
        'Validation Loss',
    ]
    archive.log(log_titles, printout=True)
    archive.create_checkpoint(epoch=0, time=0)

    # define loss function (criterion)
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = None  # TODO: define

    cudnn.benchmark = True

    # Data loading code
    training_set_path = training_set_path.expanduser()
    validation_set_path = validation_set_path.expanduser()

    transform = transforms.Compose([
        stack.ToFloatTensor(),
        # stack.RandomTranslation(2**(size-1)),
        stack.RandomRotateAndScale(),
        stack.RandomFlip(),
        stack.RandomAugmentation(),
        stack.Normalize(2)
    ])

    train_dataset = stack.compile_dataset([training_set_path], transform)
    validation_dataset = stack.compile_dataset([validation_set_path], transform)
    train_sampler = datadist.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('=========== BEGIN TRAIN LOOP ============')
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        select_params(model, optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, archive, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, archive, criterion)

        archive.update(**state_vars)
        log_vals = [
            datetime.datetime.now(),
            epoch,
            len(train_loader),
            train_loss,
            val_loss,
        ]
        archive.log(log_vals, printout=True)
        archive.create_checkpoint(epoch, time='f')


def train(train_loader, archive, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    archive.model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input = input.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        sct, tgt, truth = prepare_input(sample)

        # compute output
        output = archive.model(input)
        loss = criterion(output, tgt)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % state_vars['log_time'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            log_vals = [
               datetime.datetime.now(),
               epoch,
               i,
               loss,
               '',
            ]
            archive.log(log_vals, printout=True)
    return losses.avg


def validate(val_loader, archive, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    archive.model.eval()

    # with torch.no_grad():
    #     end = time.time()
    #     for i, (input, target) in enumerate(val_loader):
    #         input = input.cuda(args.gpu, non_blocking=True)
    #         target = target.cuda(args.gpu, non_blocking=True)

    #         # compute output
    #         output = archive.model(input)
    #         loss = criterion(output, target)

    #         # measure accuracy and record loss
    #         prec1, prec5 = accuracy(output, target, topk=(1, 5))
    #         losses.update(loss.item(), input.size(0))
    #         top1.update(prec1[0], input.size(0))
    #         top5.update(prec5[0], input.size(0))

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         if i % args.print_freq == 0:
    #             print('Test: [{0}/{1}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                    i, len(val_loader), batch_time=batch_time, loss=losses,
    #                    top1=top1, top5=top5))

    #     print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, deccay=0.1, cycle=30):
    """
    Sets the learning rate to the initial LR decayed by `deccay` every
    `cycle` epochs
    """
    lr = state_vars['start_lr'] * (deccay ** (epoch // cycle))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def select_params(model, optimizer, epoch):
    """
    Selects the parameters to be trained based on the current epoch.
    Freezes all other parameters.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_params(epoch):
        p.requires_grad = True


def select_submodule(model, optimizer, epoch):
    """
    Selects the submodule to be trained based on the current epoch.
    Freezes all other submodules of `model`.
    """
    for p in model.parameters():
        p.requires_grad = False
    submodule = model.get_submodule(epoch)
    for p in submodule.parameters():
        p.requires_grad = True
    return submodule


def prepare_input(sample):
    if state_vars['supervised']:
        src = sample['src']
        truth_field = random_field(src.shape)
        return (src,
                gridsample_residual(src, truth_field, padding_mode='zeros'),
                truth_field)
    else:
        return sample['src'], sample['tgt'], None


def random_field(shape):
    zero = torch.zeros(shape)
    warnings.warn('random_field is not implemented yet. '
                  'Using an identity field instead.')
    return zero


if __name__ == '__main__':
    main()
