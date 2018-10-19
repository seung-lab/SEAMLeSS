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
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import torchvision.transforms as transforms
from pathlib import Path

from arguments import parse_args  # TODO: move up for faster arg access
from archive import ModelArchive, warn_change
import stack_dataset as stack
from helpers import (gridsample_residual, save_chunk, dv as save_vectors, 
                     upsample, downsample)
from loss import smoothness_penalty


def main():
    global state_vars
    args = parse_args()

    # set available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    # create or load the model, optimizer, and parameters
    if args.command == 'start':
        if ModelArchive.model_exists(args.name):
            raise ValueError('The model "{}" already exists.'.format(args.name))
        archive = ModelArchive(args.name, readonly=False, seed=args.seed)
        state_vars = archive.state_vars
        state_vars['name'] = args.name
        state_vars['start_lr'] = args.lr
        state_vars['lr'] = args.lr
        state_vars['wd'] = args.wd
        state_vars['deccay'] = args.deccay
        state_vars['deccay_cycle'] = args.deccay_cycle
        state_vars['epoch'] = 0
        state_vars['num_epochs'] = args.num_epochs
        state_vars['training_set_path'] = Path(args.training_set).expanduser()
        state_vars['validation_set_path'] = Path(args.validation_set).expanduser()
        state_vars['lm'] = args.lm
        state_vars['hm'] = args.hm
        state_vars['supervised'] = args.supervised
        state_vars['batch_size'] = args.batch_size
        state_vars['log_time'] = args.log_time
        state_vars['checkpoint_time'] = args.checkpoint_time
        state_vars['vis_time'] = args.vis_time
        state_vars['lambda1'] = args.vis_time
        state_vars['penalty'] = args.penalty

        for param_group in archive.optimizer.param_groups:
            param_group['lr'] = state_vars['lr']
            param_group['weight_decay'] = state_vars['wd']
    else:  # args.command == 'resume'
        archive = ModelArchive(args.name, readonly=False)
        state_vars = archive.state_vars

        for param_group in archive.optimizer.param_groups:
            warned_lr = False
            if param_group['lr'] != state_vars['lr']:
                if not warned_lr:
                    warn_change('learning rate', param_group['lr'], state_vars['lr'])
                    warned_lr = True
                param_group['lr'] = state_vars['lr']
            warned_wd = False
            if param_group['weight_decay'] != state_vars['wd']:
                if not warned_wd:
                    warn_change('weight decay', param_group['weight_decay'], state_vars['wd'])
                    warned_wd = True
                param_group['weight_decay'] = state_vars['wd']

    # update the archive
    archive.update()
    log_titles = [
        'Time Stamp',
        'Epoch',
        'Iteration',
        'Training Loss',
        'Validation Loss',
    ]
    archive.log(log_titles, printout=False)
    archive.create_checkpoint(epoch=None, iteration=None)

    cudnn.benchmark = True

    # Data loading code
    transform = transforms.Compose([
        stack.ToFloatTensor(),
        # stack.RandomTranslation(2**(size-1)),
        stack.RandomRotateAndScale(),
        stack.RandomFlip(),
        stack.RandomAugmentation(),
        stack.Normalize(2)
    ])
    train_dataset = stack.compile_dataset([state_vars['training_set_path']], transform)
    validation_dataset = stack.compile_dataset([state_vars['validation_set_path']], transform)
    train_sampler = datadist.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('=========== BEGIN TRAIN LOOP ============')
    start_epoch = state_vars['epoch']
    for epoch in range(start_epoch, state_vars['num_epochs']):
        state_vars['epoch'] = epoch
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss = train(train_loader, archive, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, archive, epoch)

        archive.update()
        log_vals = [
            datetime.datetime.now(),
            epoch,
            len(train_loader),
            train_loss,
            val_loss,
        ]
        archive.log(log_vals, printout=True)
        archive.create_checkpoint(epoch, iteration=None)


def train(train_loader, archive, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    archive.model.train()
    archive.adjust_learning_rate()
    submodule = select_submodule(archive.model, epoch)
    submodule = torch.nn.DataParallel(submodule)

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output and loss
        stack, truth = prepare_input(sample)
        prediction = submodule(stack)
        if truth:
            loss = supervised_loss(prediction=prediction, truth=truth)
        else:
            loss = unsupervised_loss(data=stack, prediction=prediction)

        # compute gradient and do optimizer step
        archive.optimizer.zero_grad()
        loss.backward()
        archive.optimizer.step()
        loss = loss.item()  # get python value without the computation graph
        losses.update(loss)
        archive.update()

        # measure elapsed time
        batch_time.update(time.time() - end)

        # logging and checkpointing
        if i % state_vars['vis_time'] == 0:
            try:
                debug_dir = archive.new_debug_directory(epoch, i)
                save_chunk(stack[:, 0, :, :], str(debug_dir / 'src'))
                save_chunk(stack[:, 1, :, :], str(debug_dir / 'tgt'))
                save_vectors(truth, str(debug_dir / 'ground_truth'))
                save_vectors(prediction, str(debug_dir / 'prediction'))
            except Exception as e:
                # Don't raise the exception, since visualization issues
                # should not stop training. Just warn the user and go on.
                print('Visualization failed: {}'.format(e))
        if i % state_vars['checkpoint_time'] == 0:
            archive.create_checkpoint(epoch=epoch, iteration=i)
        if i % state_vars['log_time'] == 0:
            log_vals = [
               datetime.datetime.now(),
               epoch,
               i,
               loss,
               '',
            ]
            archive.log(log_vals, printout=True)
            print('Epoch: {0} [{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

        end = time.time()
    return losses.avg


def validate(val_loader, archive, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    archive.model.eval()
    submodule = select_submodule(archive.model, epoch)
    submodule = torch.nn.DataParallel(submodule)

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

    return losses.avg


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


def select_submodule(model, epoch):
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


def prepare_input(sample, max_displacement=2):
    if state_vars['supervised']:
        src = sample['src']
        truth_field = random_field(src.shape, max_displacement=max_displacement)
        tgt = gridsample_residual(src, truth_field, padding_mode='zeros')
    else:
        src = sample['src']
        tgt = sample['tgt']
        truth_field = None
    stack = torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)]).unsqueeze(0)
    return stack, truth_field


def random_field(shape, max_displacement=2):
    with torch.no_grad():
        zero = torch.zeros(shape)
        zero = torch.cat([zero, zero.clone()], 1)
        smaller = downsample(2)(zero)
        std = max_displacement / shape[-2] * math.sqrt(2)
        smaller = torch.nn.init.normal_(smaller, mean=0, std=std)
        result = upsample(2)(smaller)
    return result


def supervised_loss(prediction, truth):
    return ((prediction - truth) ** 2).mean()


def unsupervised_loss(data, prediction, src_masks=[], tgt_masks=[], field_masks=[]):
    src = data[:, 0, :, :].unsqueeze(1)
    tgt = data[:, 0, :, :].unsqueeze(1)
    src_warped = gridsample_residual(src, prediction, padding_mode='zeros')

    image_loss_map = (src_warped - tgt)**2
    image_weights = torch.ones_like(image_loss_map)
    for mask in src_masks:
        mask = gridsample_residual(mask, prediction, padding_mode='border')
        image_loss_map = image_loss_map * mask
        image_weights = image_weights * mask
    for mask in tgt_masks:
        image_loss_map = image_loss_map * mask
        image_weights = image_weights * mask
    mse_loss = image_loss_map.sum() / image_weights.sum()

    field_penalty = smoothness_penalty(state_vars['penalty'])
    field_loss_map = field_penalty([prediction])
    field_weights = torch.ones_like(prediction)
    for mask in field_masks:
        field_loss_map = field_loss_map * mask
        field_weights = field_weights * mask
    field_loss = field_loss_map.sum() / field_weights.sum()
    return mse_loss + state_vars['lambda1'] * field_loss


if __name__ == '__main__':
    main()
