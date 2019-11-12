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
import datetime
import math
import random

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchfields  # noqa: unused
from pathlib import Path

import stack_dataset
from utilities.archive import ModelArchive
from utilities.helpers import (grid_sample, save_chunk,
                               dvl as save_vectors, AverageMeter,
                               retry_enumerate, cp, dotdict)


def main():
    global state_vars
    args = parse_args()

    # set available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    # either load or create the model, optimizer, and parameters
    archive = load_archive(args)
    state_vars = archive.state_vars

    # optimize cuda processes
    cudnn.benchmark = True

    # set up training data
    train_transform = transforms.Compose([
        stack_dataset.ToFloatTensor(),
        stack_dataset.Preprocess(archive.preprocessor),
        stack_dataset.OnlyIf(stack_dataset.RandomRotateAndScale(),
                             not state_vars.skip_aug),
        stack_dataset.OnlyIf(stack_dataset.RandomFlip(),
                             not state_vars.skip_aug),
        stack_dataset.Split(),
        stack_dataset.OnlyIf(stack_dataset.RandomTranslation(20),
                             not state_vars.skip_aug),
        stack_dataset.OnlyIf(stack_dataset.RandomField(),
                             state_vars.supervised),
        stack_dataset.OnlyIf(stack_dataset.RandomAugmentation(),
                             not state_vars.skip_aug),
        stack_dataset.ToDevice('cpu'),
    ])
    train_dataset = stack_dataset.compile_dataset(
        state_vars.training_set_path, transform=train_transform,
        num_samples=state_vars.num_samples, repeats=state_vars.repeats)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=state_vars.batch_size,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        pin_memory=(state_vars.validation_set_path is None),
        sampler=train_sampler)

    # set up validation data if present
    if state_vars.validation_set_path:
        val_transform = transforms.Compose([
            stack_dataset.ToFloatTensor(),
            stack_dataset.Preprocess(archive.preprocessor),
            stack_dataset.Split(),
        ])
        validation_dataset = stack_dataset.compile_dataset(
            state_vars.validation_set_path, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=False)
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
        state_vars.levels = None
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
    submodule = select_submodule(archive.model)
    init_submodule(submodule)
    print('training levels: {}'
          .format(list(range(state_vars.height))[state_vars.levels]))

    start_time = time.time()
    start_iter = 0 if state_vars.iteration is None else state_vars.iteration
    for i, (sample, id) in retry_enumerate(train_loader, start_iter):
        if i >= len(train_loader):
            break
        state_vars.iteration = i
        sample = dotdict(sample)
        id = id[0].item()

        # measure data loading time
        data_time.update(time.time() - start_time)

        # compute output and loss
        if torch.cuda.device_count() == 1:
            sample = stack_dataset.ToDevice('cuda')(sample)
        src = sample.src.image if sample.src.aug is None else sample.src.aug
        tgt = sample.tgt.image if sample.tgt.aug is None else sample.tgt.aug
        prediction = submodule(src, tgt)
        loss = archive.loss(sample, prediction=prediction)
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
                and i % state_vars.checkpoint_time == 0 and i != 0):
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
                  'Sample: {id}\t'
                  'Loss {loss.val:12.10f} ({loss.avg:.10f})\t'
                  'BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  .format(
                      state_vars.name,
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, id=id))
        if state_vars.vis_time and i % state_vars.vis_time == 0:
            create_debug_outputs(archive, sample, prediction, id)
        elif i % 50:
            archive.visualize_loss('Training Loss', 'Validation Loss')

        start_time = time.time()
    return losses.avg


@torch.no_grad()
def validate(val_loader, archive, epoch):
    losses = AverageMeter()

    # switch to evaluate mode
    archive.model.eval()
    submodule = select_submodule(archive.model)

    # compute output and loss
    start_time = time.time()
    for i, (sample, id) in retry_enumerate(val_loader):
        sample = dotdict(sample)
        print('{0}\t'
              'Validation: [{1}/{2}]\t'
              .format(state_vars.name, i, len(val_loader)), end='\r')
        prediction = submodule(sample.src.image, sample.tgt.image)
        loss = archive.val_loss(sample, prediction=prediction)
        losses.update(loss.item())

    # measure elapsed time
    batch_time = (time.time() - start_time)

    # debugging outputs and printing
    create_debug_outputs(archive, sample, prediction)
    print('{0}\t'
          'Validation: [{1}/{1}]\t'
          'Loss {loss.avg:.10f}\t\t\t'
          'Time {batch_time:.3f}\t'
          .format(state_vars.name, len(val_loader),
                  batch_time=batch_time, loss=losses))

    return losses.avg


def select_submodule(model):
    """
    Selects the submodule to be trained based on the specified levels.
    If `levels` is `None`, selects them by calling `select_levels()`
    """
    if state_vars.levels is None:
        state_vars.levels = select_levels()
    submodule = model.module[state_vars.levels]
    if state_vars.plan == 'top_down':
        last = 'lowest'
    elif state_vars.plan == 'bottom_up':
        last = 'highest'
    else:
        last = 'all'
    return torch.nn.DataParallel(submodule.train_level(last))


def select_levels():
    """
    Returns a slice, list, or integer representing the levels to be trained
    during this epoch.
    At epoch `epoch`, selects version `epoch/epochs_per_mip` of the model,
    according to the training plan `plan`.
    `plan` may be any of `all`, `top_down`, `bottom_up`, or `random_one`
    """
    if state_vars.epoch is None:
        return slice(None)
    index = state_vars.epoch // state_vars.epochs_per_mip
    if state_vars.plan == 'top_down':
        return slice(-(index+1), None)
    elif state_vars.plan == 'bottom_up':
        return slice(None, index+1)
    elif state_vars.plan == 'random_one':
        level = random.randrange(0, state_vars.height + 1)
        return level if level < state_vars.height else slice(None)
    else:  # state_vars.plan == 'all':
        return slice(None)


def init_submodule(submodule):
    """
    Initializes the last level of the submodule to the weights of the
    next-to-last, if this has not already happened.
    """
    index = state_vars.epoch // state_vars.epochs_per_mip
    if state_vars.initialized_list is None:
        state_vars.initialized_list = []
    if len(state_vars.initialized_list) < state_vars.height:
        state_vars.initialized_list += \
            [False]*(state_vars.height - len(state_vars.initialized_list))
    if (index < state_vars.height
            and not state_vars.initialized_list[index]):
        if state_vars.plan == 'top_down':
            submodule.module.init_level('lowest')
        elif state_vars.plan == 'bottom_up':
            submodule.module.init_level('highest')
        state_vars.initialized_list[index] = True
    return submodule


@torch.no_grad()
def create_debug_outputs(archive, sample, prediction, id=0):
    """
    Creates a subdirectory exports any debugging outputs to that directory.
    """
    try:
        debug_dir = archive.new_debug_directory(exist_ok=True)
        stack_dir = debug_dir / 'stack'
        stack_dir.mkdir(exist_ok=True)
        sample = stack_dataset.ToDevice('cuda')(sample)
        src, tgt = sample.src.image, sample.tgt.image
        save_chunk(src[0:1, ...], str(debug_dir / 'src_{}'.format(id)))
        # cp(debug_dir / 'src_{}.png'.format(id), stack_dir)
        save_chunk(tgt[0:1, ...], str(debug_dir / 'tgt_{}'.format(id)))
        # cp(debug_dir / 'tgt_{}.png'.format(id), stack_dir)
        src_aug, tgt_aug = sample.src.aug, sample.tgt.aug
        if src_aug is not None:
            save_chunk(src_aug[0:1, ...],
                       str(debug_dir / 'src_aug_{}'.format(id)))
            cp(debug_dir / 'src_aug_{}.png'.format(id), stack_dir)
        else:
            cp(debug_dir / 'src_{}.png'.format(id), stack_dir)
        if tgt_aug is not None:
            save_chunk(tgt_aug[0:1, ...],
                       str(debug_dir / 'tgt_aug_{}'.format(id)))
            cp(debug_dir / 'tgt_aug_{}.png'.format(id), stack_dir)
        else:
            cp(debug_dir / 'tgt_{}.png'.format(id), stack_dir)
        prediction = prediction.field_()
        warped_src = prediction[0:1, ...].detach().to(src.device).sample(
            src[0:1, ...])
        save_chunk(warped_src[0:1, ...], str(debug_dir / 'warped_src'))
        cp(debug_dir / 'warped_src.png', stack_dir)
        archive.visualize_loss('Training Loss', 'Validation Loss')
        cp(archive.paths['plot'], debug_dir)  # make copy of the training curve
        save_vectors(prediction[0:1, ...].detach().permute(0, 2, 3, 1),
                     str(debug_dir / 'prediction'), mag=30)
        save_chunk(prediction[0:1, ...].detach().magnitude(keepdim=True),
                   str(debug_dir / 'prediction_img'), norm=False)
        if 'image_loss_map' in sample:
            save_chunk(sample.image_loss_map, str(debug_dir/'image_loss_map'))
        if 'field_loss_map' in sample:
            save_chunk(sample.field_loss_map, str(debug_dir/'field_loss_map'))
        if 'truth' in sample:
            save_vectors(sample.truth[0:1, ...].detach().permute(0, 2, 3, 1),
                         str(debug_dir / 'ground_truth'), mag=30)
        masks = archive._objective.prepare_masks(sample)
        for k, v in masks.items():
            if v is not None and len(v) > 0:
                for i in range(len(v)):
                    save_chunk(v[i][0:1, ...],
                               str(debug_dir / '{}_{}'.format(k, i)))
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
            archive = ModelArchive.start_new(old, **vars(args))
            # TODO: explicitly remove old model from memory
        else:
            archive = ModelArchive(readonly=False, **vars(args))
            archive.set_log_titles([
                'Time Stamp',
                'Epoch',
                'Iteration',
                'Training Loss',
                'Validation Loss',
            ])
        archive.state_vars.update({
            'name': args.name,
            'height': args.height,
            'feature_maps': args.feature_maps,
            'start_lr': args.lr,
            'lr': args.lr,
            'wd': args.wd,
            'gamma': args.gamma,
            'gamma_step': args.gamma_step,
            'epoch': 0,
            'iteration': None,
            'num_epochs': args.num_epochs,
            'plan': args.plan,
            'epochs_per_mip': args.epochs_per_mip,
            'training_set_path': Path(args.training_set).expanduser(),
            'validation_set_path':
                Path(args.validation_set).expanduser() if args.validation_set
                else None,
            'skip_aug': args.skip_aug,
            'num_samples': args.num_samples,
            'repeats': args.repeats,
            'supervised': args.supervised,
            'encodings': args.encodings,
            'batch_size': args.batch_size,
            'log_time': args.log_time,
            'checkpoint_time': args.checkpoint_time,
            'vis_time': args.vis_time,
            'lambda1': args.lambda1,
            'penalty': args.penalty,
            'gpus': args.gpu_ids,
        })
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
        archive.state_vars['batch_size'] = args.batch_size
        archive.state_vars['repeats'] = args.repeats

    # record a training session
    archive.record_training_session()

    # redirect output through the archive
    sys.stdout = archive.out
    sys.stderr = archive.err

    return archive


if __name__ == '__main__':
    main()
