#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Train a multilayer aligner network.

This is the main module which is invoked to train a network.

Running:
    To begin training, run

        $ python3 train.py start MODEL_NAME --training_set SET [...]

    or equivalently

        $ ./train.py start MODEL_NAME --training_set SET [...]

    To get help with the command line options, use

        $ python3 train.py start --help

Resuming:

    It is possible to resume training. This is useful if a training run was
    killed by accident or circumstance and you would like to continue training.
    Before doing this, check to make sure the run is actually in fact dead,
    since attempting to resume a live run could have undefined behavior.
    To resume training run the following with no additional command line
    arguments:

        $ python3 train.py resume MODEL_NAME

    and where `MODEL_NAME` is the name of the previously stopped training run.
    The training parameters and training state will be loaded from the saved
    archive, and the model will continue to train from where it was stopped.

Example:
        $ python3 train.py start my_model --training_set training_data.h5

Specifying the GPUs:

    If not specified explicitly, the first avalailable unused GPU will be
    selected (or the least used GPU if all are in use).
    If you would like to use a specific GPU, or multiple GPUs, use the
    `--gpu_ids` argument with a comma-separated list of IDs:

        $ python3 train.py --gpu_ids 4,1,2 start my_model \
            --training_set training_data.h5

    The order maters insomuch as the first ID in the list will be the
    default GPU, and therefore will generally experience higher usage
    than the others.
    So the above command will use GPUs 1, 2, and 4, with 4 as the default.

    Note that this must come before the `start` or `resume` command,
    and must be specified again (if desired) upon resuming:

        $ python3 train.py --gpu_ids 5,3,6 resume my_model

    The reason for this is that the model may resume training on different
    GPUs, or even on a different machine, than where it started its training.

Editor's note:

    The top of this file should contain the comment `# PYTHON_ARGCOMPLETE_OK`
    in order to allow command line tab completion to work properly.

"""
from arguments import parse_args  # keep first for fast args access

import os
import sys
import time
import warnings
import datetime
import math
import random
import math

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from pathlib import Path

import masks as masklib
import stack_dataset
from utilities.archive import ModelArchive
from utilities.helpers import (gridsample_residual, save_chunk,
                               dvl as save_vectors,
                               upsample, downsample, AverageMeter,
                               retry_enumerate)


def main():
    global state_vars
    args = parse_args()

    # set available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # either load or create the model, optimizer, and parameters
    archive = load_archive(args)
    state_vars = archive.state_vars

    # optimize cuda processes
    cudnn.benchmark = True

    # set up training data
    train_transform = transforms.Compose([
        stack_dataset.ToFloatTensor(),
        archive.preprocessor,
        stack_dataset.RandomRotateAndScale(),
        stack_dataset.RandomFlip(),
        stack_dataset.Split(),
    ])
    train_dataset = stack_dataset.compile_dataset(
        state_vars.training_set_path, transform=train_transform)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=state_vars.batch_size,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        pin_memory=True, sampler=train_sampler)

    # set up validation data if present
    if state_vars.validation_set_path:
        val_transform = transforms.Compose([
            stack_dataset.ToFloatTensor(),
            archive.preprocessor,
            stack_dataset.RandomRotateAndScale(),
            stack_dataset.RandomFlip(),
            stack_dataset.Split(),
        ])
        validation_dataset = stack_dataset.compile_dataset(
            state_vars.validation_set_path, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)
    else:
        val_loader = None

    # Averaging
    train_losses = AverageMeter()
    val_losses = AverageMeter()
    epoch_time = AverageMeter()

    print('=========== BEGIN TRAIN LOOP ============')
    start_epoch = state_vars.epoch
    for epoch in range(start_epoch, state_vars.num_epochs):
        start_time = time.time()
        state_vars.epoch = epoch
        archive.save()

        # train for one epoch
        train_loss = train(train_loader, archive, epoch)
        train_losses.update(train_loss)

        # evaluate on validation set
        if val_loader:
            val_loss = validate(val_loader, archive, epoch)
            val_losses.update(val_loss)
        else:
            val_loss = None

        # log and save state
        state_vars.epoch = epoch + 1
        state_vars.iteration = None
        archive.save()
        archive.log([
            datetime.datetime.now(),
            epoch,
            len(train_loader),
            '',  # train_loss
            val_loss if val_loss is not None else '',
        ])
        archive.create_checkpoint(epoch, iteration=None)
        epoch_time.update(time.time() - start_time)
        print('{0}\t'
              'Completed Epoch {1}\t'
              'TrainLoss {train_losses.val:.10f} ({train_losses.avg:.10f})\t'
              'ValLoss {val_losses.val:.10f} ({val_losses.avg:.10f})\t'
              'EpochTime {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t'
              '\n'
              .format(state_vars.name, epoch, train_losses=train_losses,
                      val_losses=val_losses, epoch_time=epoch_time))


def train(train_loader, archive, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode and select the submodule to train
    archive.model.train()
    archive.adjust_learning_rate()
    submodule = select_submodule(archive.model, epoch, init=True)
    max_disp = submodule.module.pixel_size_ratio * 2  # correct 2-pixel disp

    start_time = time.time()
    start_iter = 0 if state_vars.iteration is None else state_vars.iteration
    for i, sample in retry_enumerate(train_loader, start_iter):
        if i >= len(train_loader):
            break
        state_vars.iteration = i

        # measure data loading time
        data_time.update(time.time() - start_time)

        # compute output and loss
        src, tgt, truth = prepare_input(sample, max_displacement=max_disp)
        prediction = submodule(src, tgt)
        masks = gen_masks(src, tgt, prediction)
        if truth is not None:
            loss = archive.loss(prediction=prediction, truth=truth, **masks)
        else:
            loss = archive.loss(src, tgt, prediction=prediction, **masks)
        loss = loss.mean()  # average across a batch if present

        # compute gradient and do optimizer step
        if math.isfinite(loss.item()):
            archive.optimizer.zero_grad()
            loss.backward()
            archive.optimizer.step()
        loss = loss.item()  # get python value without the computation graph
        losses.update(loss)
        state_vars.iteration = i + 1  # advance iteration to resume correctly
        archive.save()
        state_vars.iteration = i  # revert back for logging & visualizations

        # measure elapsed time
        batch_time.update(time.time() - start_time)

        # debugging, logging, and checkpointing
        if (state_vars.checkpoint_time
                and i % state_vars.checkpoint_time == 0):
            archive.create_checkpoint(epoch=epoch, iteration=i)
        if state_vars.log_time and i % state_vars.log_time == 0:
            archive.log([
               datetime.datetime.now(),
               epoch,
               i,
               loss if math.isfinite(loss) else '',
               '',
            ])
            print('{0}\t'
                  'Epoch: {1} [{2}/{3}]\t'
                  'Loss {loss.val:12.10f} ({loss.avg:.10f})\t'
                  'BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  .format(
                      state_vars.name,
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
        if state_vars.vis_time and i % state_vars.vis_time == 0:
            create_debug_outputs(archive, src, tgt, prediction, truth, masks)

        start_time = time.time()
    return losses.avg


@torch.no_grad()
def validate(val_loader, archive, epoch):
    losses = AverageMeter()

    # switch to evaluate mode
    archive.model.eval()
    submodule = select_submodule(archive.model, epoch)

    # compute output and loss
    start_time = time.time()
    for i, sample in retry_enumerate(val_loader):
        print('{0}\t'
              'Validation: [{1}/{2}]\t'
              .format(state_vars.name, i, len(val_loader)), end='\r')
        src, tgt, truth = prepare_input(sample, supervised=False)
        prediction = submodule(src, tgt)
        masks = gen_masks(src, tgt, prediction)
        loss = archive.val_loss(src, tgt, prediction=prediction, **masks)
        losses.update(loss.item())

    # measure elapsed time
    batch_time = (time.time() - start_time)

    # debugging outputs and printing
    create_debug_outputs(archive, src, tgt, prediction, truth, masks)
    print('{0}\t'
          'Validation: [{1}/{1}]\t'
          'Loss {loss.avg:.10f}\t\t\t'
          'Time {batch_time:.3f}\t'
          .format(state_vars.name, len(val_loader),
                  batch_time=batch_time, loss=losses))

    return losses.avg


def select_submodule(model, epoch, init=False):
    """
    Selects the submodule to be trained based on the current epoch.
    At epoch `epoch`, train level `epoch/epochs_per_mip` of the model.
    """
    if epoch is None:
        return model
    index = epoch // state_vars.epochs_per_mip
    submodule = model.module[:index+1].train_last()
    if (init and epoch % state_vars.epochs_per_mip == 0
            and index < state_vars.height
            and state_vars.iteration is None):
        submodule.init_last()
    return torch.nn.DataParallel(submodule)


@torch.no_grad()
def prepare_input(sample, supervised=None, max_displacement=2):
    """
    Formats the input received from the data loader and produces a
    ground truth vector field if supervised.
    If `supervised` is None, it uses the value specified in state_vars
    """
    if supervised is None:
        supervised = state_vars.supervised
    if supervised:
        src = sample['src'].cuda()
        truth_field = random_field(src.shape, max_displacement=max_displacement)
        tgt = gridsample_residual(src, truth_field, padding_mode='zeros')
    else:
        src = sample['src'].cuda()
        tgt = sample['tgt'].cuda()
        truth_field = None
    return src, tgt, truth_field


@torch.no_grad()
def random_field(shape, max_displacement=2, num_downsamples=7):
    """
    Genenerates a random vector field smoothed by bilinear interpolation.

    The vectors generated will have values representing displacements of
    between (approximately) `-max_displacement` and `max_displacement` pixels
    at the size dictated by `shape`.
    The actual values, however, will be scaled to the spatial transformer
    standard, where -1 and 1 represent the edges of the image.

    `num_downsamples` dictates the block size for the random field.
    Each block will have size `2**num_downsamples`.
    """
    zero = torch.zeros(shape, device='cuda')
    zero = torch.cat([zero, zero.clone()], 1)
    smaller = downsample(num_downsamples)(zero)
    std = max_displacement / shape[-2] / math.sqrt(2)
    field = torch.nn.init.normal_(smaller, mean=0, std=std)
    field = upsample(num_downsamples)(field)
    result = field.permute(0, 2, 3, 1)
    return result


@torch.no_grad()
def gen_masks(src, tgt, prediction=None, threshold=10):
    """
    Returns masks with which to weight the loss function
    """
    if prediction is not None:
        src, tgt = src.to(prediction.device), tgt.to(prediction.device)
    src, tgt = (src * 255).to(torch.uint8), (tgt * 255).to(torch.uint8)

    src_mask, tgt_mask = torch.ones_like(src), torch.ones_like(tgt)

    src_mask_zero, tgt_mask_zero = (src < threshold), (tgt < threshold)
    src_mask_five = masklib.dilate(src_mask_zero, radius=3)
    tgt_mask_five = masklib.dilate(tgt_mask_zero, radius=3)
    src_mask[src_mask_five], tgt_mask[tgt_mask_five] = 5, 5
    src_mask[src_mask_zero], tgt_mask[tgt_mask_zero] = 0, 0

    src_field_mask, tgt_field_mask = torch.ones_like(src), torch.ones_like(tgt)
    src_field_mask[src_mask_zero], tgt_field_mask[tgt_mask_zero] = 0, 0

    return {'src_masks': [src_mask.float()],
            'tgt_masks': [tgt_mask.float()],
            'src_field_masks': [src_field_mask.float()],
            'tgt_field_masks': [tgt_field_mask.float()]}


@torch.no_grad()
def create_debug_outputs(archive, src, tgt, prediction, truth, masks):
    """
    Creates a subdirectory exports any debugging outputs to that directory.
    """
    try:
        debug_dir = archive.new_debug_directory()
        save_chunk(src[0:1, ...], str(debug_dir / 'src'))
        save_chunk(src[0:1, ...], str(debug_dir / 'xsrc'))  # extra copy of src
        save_chunk(tgt[0:1, ...], str(debug_dir / 'tgt'))
        warped_src = gridsample_residual(
            src[0:1, ...],
            prediction[0:1, ...].detach().to(src.device),
            padding_mode='zeros')
        save_chunk(warped_src[0:1, ...], str(debug_dir / 'warped_src'))
        archive.visualize_loss('Training Loss', 'Validation Loss')
        save_vectors(prediction[0:1, ...].detach(),
                     str(debug_dir / 'prediction'))
        if truth is not None:
            save_vectors(truth[0:1, ...].detach(),
                         str(debug_dir / 'ground_truth'))
        for k, v in masks.items():
            if v is not None and len(v) > 0:
                save_chunk(v[0][0:1, ...], str(debug_dir / k))
    except Exception as e:
        # Don't raise the exception, since visualization issues
        # should not stop training. Just warn the user and go on.
        print('Visualization failed: {}: {}'.format(e.__class__.__name__, e))


def load_archive(args):
    """
    Load or create the model, optimizer, and parameters as a `ModelArchive`.

    If the command is `start`, this attempts to create a new `ModelArchive`
    with name `args.name`, if possible (that is, without overwriting an
    existing one).
    If the command is `resume`, this attempts to load it from disk.
    """
    if args.command == 'start':
        if ModelArchive.model_exists(args.name):
            raise ValueError('The model "{}" already exists.'
                             .format(args.name))
        if args.saved_model is not None:
            # load a previous model and create a copy
            if not ModelArchive.model_exists(args.saved_model):
                raise ValueError('The model "{}" could not be found.'
                                 .format(args.saved_model))
            old = ModelArchive(args.saved_model, readonly=True)
            archive = old.start_new(readonly=False, **vars(args))
            # TODO: explicitly remove old model from memory
        else:
            archive = ModelArchive(readonly=False, **vars(args))
        archive.state_vars.update({
            'name': args.name,
            'height': args.height,
            'start_lr': args.lr,
            'lr': args.lr,
            'wd': args.wd,
            'gamma': args.gamma,
            'gamma_step': args.gamma_step,
            'epoch': 0,
            'iteration': None,
            'num_epochs': args.num_epochs,
            'epochs_per_mip': args.epochs_per_mip,
            'training_set_path': Path(args.training_set).expanduser(),
            'validation_set_path':
                Path(args.validation_set).expanduser() if args.validation_set
                else None,
            'supervised': args.supervised,
            'batch_size': args.batch_size,
            'log_time': args.log_time,
            'checkpoint_time': args.checkpoint_time,
            'vis_time': args.vis_time,
            'lambda1': args.lambda1,
            'penalty': args.penalty,
            'gpus': args.gpu_ids,
        })
        archive.set_log_titles([
            'Time Stamp',
            'Epoch',
            'Iteration',
            'Training Loss',
            'Validation Loss',
        ])
        archive.set_optimizer_params(learning_rate=args.lr,
                                     weight_decay=args.wd)

        # save initialized state to archive; create first checkpoint
        archive.save()
        archive.create_checkpoint(epoch=None, iteration=None)
    else:  # args.command == 'resume'
        if not ModelArchive.model_exists(args.name):
            raise ValueError('The model "{}" could not be found.'
                             .format(args.name))
        archive = ModelArchive(args.name, readonly=False)
        archive.state_vars['gpus'] = args.gpu_ids

    # redirect output through the archive
    sys.stdout = archive.out
    sys.stderr = archive.err

    return archive


if __name__ == '__main__':
    main()
