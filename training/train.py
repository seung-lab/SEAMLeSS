"""Train a network.

This is the main module which is invoked to train a network.

Running:
    To begin training, run this on a machine with a GPU.
    (Don't forget to install the required dependencies in requirements.txt)

        $ python3 train.py [--param1_name VALUE1 --param2_name VALUE2 ...]
            EXPERIMENT_NAME

Example:
        $ python3 train.py --state_archive pt/SOME_ARCHIVE.pt --size 8
            --lambda1 2 --lambda2 0.04 --lambda3 0 --lambda4 5 --lambda5 0
            --mask_smooth_radius 75 --mask_neighborhood_radius 75 --lr 0.0003
            --trunc 0 --fine_tuning --hm --padding 0 --vis_interval 5
            --lambda6 1 fine_tune_example

Todo:
    * Refactor main method
    * Expand docstring

"""

import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import random

import masks
from stack_dataset import StackDataset
from pyramid import PyramidTransformer
from defect_net import *
from defect_detector import DefectDetector
from helpers import reverse_dim, downsample
from aug import aug_stacks, aug_input, rotate_and_scale, crack, displace_slice
from vis import visualize_outputs
from loss import similarity_score, smoothness_penalty
from normalizer import Normalizer


def main():
    """Start training a net."""
    args = parse_args()
    name = args.name
    trunclayer = args.trunc
    skiplayers = args.skip
    size = args.size
    fall_time = args.fall_time
    padding = args.padding
    dim = args.dim + padding
    kernel_size = args.k
    log_path = 'out/' + name + '/'
    log_file = log_path + name + '.log'
    lr = args.lr
    # batch_size = args.batch_size
    fine_tuning = args.fine_tuning
    epoch = args.epoch
    start_time = time.time()
    similarity = similarity_score(should_reduce=False)
    smoothness = smoothness_penalty(args.penalty)
    history = []

    # setup the defect detector
    if args.mm or args.hm:
        hm_defect_detector = DefectDetector(
            torch.load(args.hm_defect_net).cpu().cuda(),
            major_dilation_radius=args.mask_neighborhood_radius,
            sigmoid_threshold = 0.5 if args.hm else 0.4)
    if args.mm or (not args.hm):
        lm_defect_detector = DefectDetector(
            torch.load(args.lm_defect_net).cpu().cuda(),
            major_dilation_radius=args.mask_neighborhood_radius,
            sigmoid_threshold = 0.5 if args.hm else 0.4)
    normalizer = Normalizer(5 if args.hm else 2)

    if args.mm:
        train_dataset1 = StackDataset(os.path.expanduser(args.hm_src), mip=5)
        train_dataset2 = StackDataset(os.path.expanduser(args.lm_src), mip=2)
        train_dataset3 = StackDataset(os.path.expanduser(args.vhm_src), mip=8)
        train_dataset = ConcatDataset(
            [train_dataset1, train_dataset2, train_dataset3])
    else:
        if args.hm:
            train_dataset1 = StackDataset(
                os.path.expanduser(args.hm_src), mip=5)
            train_dataset2 = StackDataset(
                os.path.expanduser(args.vhm_src), mip=8)
            train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        else:
            train_dataset2 = StackDataset(  # dataset focused on extreme folds
                os.path.expanduser(args.lm_src), mip=2)
            train_dataset = ConcatDataset([train_dataset2])
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=5,
        pin_memory=True)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir('pt'):
        os.makedirs('pt')

    with open(log_path + 'args.txt', 'a') as f:
        f.write(str(args))

    if args.state_archive is None:
        model = PyramidTransformer(
            size=size, dim=dim, skip=skiplayers, k=kernel_size).cuda()
    else:
        model = PyramidTransformer.load(
            args.state_archive, height=size, dim=dim, skips=skiplayers,
            k=kernel_size)
        for p in model.parameters():
            p.requires_grad = True
        model.train(True)

    def opt(layer):
        params = []
        if not args.pe_only and not args.enc_only:
            if layer >= skiplayers and not fine_tuning:
                print('Training only layer {}'.format(layer))
                params.extend(model.pyramid.mlist[layer].parameters())
            else:
                print('Training all residual networks.')
                params.extend(model.pyramid.mlist.parameters())

            if fine_tuning or epoch < fall_time - 1:
                print('Training all encoder parameters.')
                params.extend(model.pyramid.enclist.parameters())
                params.extend(model.pyramid.pe.parameters())
            else:
                print('Freezing encoder parameters.')
        elif args.pe_only:
            print('Training pre-encoder only.')
            params.extend(model.pyramid.pe.parameters())
        elif args.enc_only:
            print('Training all encoder parameters only.')
            params.extend(model.pyramid.enclist.parameters())
            params.extend(model.pyramid.pe.parameters())

        lr_ = lr if not fine_tuning else lr * args.fine_tuning_lr_factor
        print(
            'Building optimizer for layer {} (fine tuning: {}, lr: {})'
            .format(layer, fine_tuning, lr_))
        return torch.optim.Adam(
            params, lr=lr_, weight_decay=0.0001 if args.pe_only else 0)

    def prefix(tag=None):
        if tag is None:
            return '{}{}_e{}_t{}_'.format(log_path, name, epoch, t)
        else:
            return '{}{}_e{}_t{}_{}_'.format(log_path, name, epoch, t, tag)

    def run_pair(src, target, src_mask=None, target_mask=None, train=True):
        if train and not args.skip_sample_aug:
            # random rotation
            should_rotate = random.randint(0, 1) == 0
            if should_rotate:
                src, grid = rotate_and_scale(
                    src.unsqueeze(0).unsqueeze(0), None)
                target = rotate_and_scale(
                    target.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                if src_mask is not None:
                    src_mask = torch.ceil(
                        rotate_and_scale(
                            src_mask.unsqueeze(0).unsqueeze(0), grid=grid
                        )[0].squeeze())
                    target_mask = torch.ceil(
                        rotate_and_scale(
                            target_mask.unsqueeze(0).unsqueeze(0), grid=grid
                        )[0].squeeze())

                src = src.squeeze()

        input_src = src.clone()
        input_target = target.clone()

        src_cutout_masks = []
        target_cutout_masks = []

        if train and not args.skip_sample_aug:
            input_src, src_cutout_masks = aug_input(input_src)
            input_target, target_cutout_masks = aug_input(input_target)
        elif train:
            print('Skipping sample-wise augmentation.')

        src = Variable(torch.FloatTensor(normalizer.apply(
            src.data.cpu().numpy()))).cuda()
        target = Variable(torch.FloatTensor(normalizer.apply(
            target.data.cpu().numpy()))).cuda()
        input_src = Variable(torch.FloatTensor(normalizer.apply(
            input_src.data.cpu().numpy()))).cuda()
        input_target = Variable(torch.FloatTensor(normalizer.apply(
            input_target.data.cpu().numpy()))).cuda()

        rf = run_sample(
            src, target, input_src, input_target, src_mask, target_mask,
            src_cutout_masks, target_cutout_masks, train)
        if rf is not None:
            if not args.pe_only:
                cost = (rf['similarity_error']
                        + args.lambda1
                        * smooth_factor * rf['smoothness_error'])
                (cost/2).backward(retain_graph=args.lambda6 > 0)
            else:
                (rf['contrast_error']/2).backward()

        src = reverse_dim(reverse_dim(src, 0), 1)
        target = reverse_dim(reverse_dim(target, 0), 1)
        input_src = reverse_dim(reverse_dim(input_src, 0), 1)
        input_target = reverse_dim(reverse_dim(input_target, 0), 1)
        src_mask = reverse_dim(reverse_dim(src_mask, 0), 1)
        target_mask = reverse_dim(reverse_dim(target_mask, 0), 1)
        src_cutout_masks = [reverse_dim(reverse_dim(m, -2), -1)
                            for m in src_cutout_masks]
        target_cutout_masks = [reverse_dim(reverse_dim(m, -2), -1)
                               for m in target_cutout_masks]

        rb = run_sample(
            src, target, input_src, input_target, src_mask, target_mask,
            src_cutout_masks, target_cutout_masks, train)
        if rb is not None:
            if not args.pe_only:
                cost = (rb['similarity_error']
                        + args.lambda1
                        * smooth_factor * rb['smoothness_error'])
                (cost/2).backward(retain_graph=args.lambda6 > 0)
            else:
                (rb['contrast_error']/2).backward()

        ##################################
        # CONSENSUS PENALTY COMPUTATION ##
        ##################################
        if (not args.pe_only and args.lambda6 > 0 and rf is not None
                and rb is not None):
            zmf = (rf['pred'] != 0).float() * (rf['input_target'] != 0).float()
            zmf = (zmf.detach().unsqueeze(0).view(1, 1, dim, dim)
                   .repeat(1, 2, 1, 1).permute(0, 2, 3, 1))
            zmb = (rb['pred'] != 0).float() * (rb['input_target'] != 0).float()
            zmb = (zmb.detach().unsqueeze(0).view(1, 1, dim, dim)
                   .repeat(1, 2, 1, 1).permute(0, 2, 3, 1))
            rffield, rbfield = rf['rfield'] * zmf, rb['rfield'] * zmb
            rbfield_reversed = -reverse_dim(reverse_dim(rbfield, 1), 2)
            consensus_diff = rffield - rbfield_reversed
            consensus_error_field = (consensus_diff[:, :, :, 0] ** 2
                                     + consensus_diff[:, :, :, 1] ** 2)
            mean_consensus = torch.mean(consensus_error_field)
            rf['consensus_field'] = rffield
            rb['consensus_field'] = rbfield_reversed
            rf['consensus_error_field'] = consensus_error_field
            rb['consensus_error_field'] = consensus_error_field
            rf['consensus'] = mean_consensus
            rb['consensus'] = mean_consensus
            consensus = args.lambda6 * mean_consensus
            consensus.backward()

        return rf, rb

    def run_sample(src, target, input_src, input_target, mask=None,
                   target_mask=None, src_cutout_masks=[],
                   target_cutout_masks=[], train=True):
        model.train(train)
        if args.dry_run:
            print('Bailing for dry run [not an error].')
            return None

        # if we're just training the pre-encoder, no need to do everything
        if args.pe_only:
            contrasted_stack = model.apply(input_src, input_target, pe=True)
            pred = contrasted_stack.squeeze()
            y = torch.cat((src.unsqueeze(0), target.squeeze().unsqueeze(0)), 0)
            contrast_err = similarity(pred, y)
            contrast_err_src = contrast_err[0:1]
            contrast_err_target = contrast_err[1:2]
            for m in src_cutout_masks:
                contrast_err_src = (contrast_err_src
                                    * m.view(contrast_err_src.size()).float())
            for m in target_cutout_masks:
                contrast_err_target = (contrast_err_target
                                       * m.view(contrast_err_target.size())
                                       .float())
            contrast_err = torch.mean(torch.cat((contrast_err_src,
                                                 contrast_err_target), 0))
            return {'contrast_error': contrast_err}

        pred, field, residuals = model.apply(input_src, input_target,
                                             trunclayer)
        # resample tensors that are in our source coordinate space with our
        # new field prediction to move them into target coordinate space so
        # we can compare things fairly
        pred = F.grid_sample(
            downsample(trunclayer)(src.unsqueeze(0).unsqueeze(0)), field)
        raw_mask = mask
        if mask is not None:
            mask = torch.ceil(
                F.grid_sample(
                    downsample(trunclayer)(mask.unsqueeze(0).unsqueeze(0)),
                    field))
        if target_mask is not None:
            target_mask = masks.dilate(
                target_mask.unsqueeze(0).unsqueeze(0), 1, binary=False)
        if len(src_cutout_masks) > 0:
            src_cutout_masks = [
                torch.ceil(F.grid_sample(m.float(), field)).byte()
                for m in src_cutout_masks]

        # first we'll build a binary mask to completely ignore
        # 'irrelevant' pixels
        similarity_binary_masks = []

        # mask away similarity error from pixels outside the boundary of
        # either prediction or target
        # in other words, we need valid data in both the target and prediction
        # to count the error from that pixel
        border_mask = masks.low_pass(masks.intersect([pred != 0, target != 0]))
        border_mask, no_valid_pixels = masks.contract(border_mask, 5,
                                                      return_sum=True)
        if no_valid_pixels:
            print('Skipping empty border mask.')
            visualize_outputs(prefix('empty_border_mask') + '{}',
                              {'src': input_src, 'target': input_target})
            return None
        similarity_binary_masks.append(border_mask)

        # mask away similarity error from pixels within cutouts in either the
        # source or target
        cutout_similarity_masks = src_cutout_masks + target_cutout_masks
        if len(cutout_similarity_masks) > 0:
            cutout_similarity_mask = masks.union(cutout_similarity_masks)
            cutout_similarity_mask = masks.invert(cutout_similarity_mask)
            similarity_binary_masks.append(cutout_similarity_mask)

        # mask away similarity error for pixels within a defect in the
        # target image
        if target_mask is not None:
            target_defect_similarity_mask = masks.contract(
                target_mask < 2, 2).detach()
            similarity_binary_masks.append(target_defect_similarity_mask)

        similarity_binary_mask = masks.intersect(similarity_binary_masks)

        # reweight remaining pixels for focus areas
        similarity_weights = Variable(torch.ones(border_mask.size())).cuda()
        if mask is not None:
            similarity_weights.data[
                masks.intersect([mask > 0, mask <= 1]).data
            ] = args.lambda4
            similarity_weights.data[
                masks.dilate(mask > 1, 2).data
            ] = args.lambda5

        similarity_weights *= similarity_binary_mask.float()

        if torch.sum(similarity_weights[border_mask.data]).data[0] < args.eps:
            print('Skipping all zero similarity weights (factor == 0).')
            visualize_outputs(prefix('zero_similarity_weights') + '{}',
                              {'src': input_src, 'target': input_target})
            return None

        # similarity_mask_factor = (
        #     torch.sum(border_mask.float())
        #     / torch.sum(similarity_weights[border_mask.data])).data[0]
        similarity_mask_factor = 1

        # compute raw similarity error and masked/weighted similarity error
        similarity_error_field = similarity(pred, target)
        weighted_similarity_error_field = (similarity_error_field
                                           * similarity_weights
                                           * similarity_mask_factor)

        # perform masking/weighting for smoothness penalty
        smoothness_binary_mask = (masks.intersect([src != 0, target != 0])
                                  .view(similarity_binary_mask.size()))
        smoothness_weights = Variable(
            torch.ones(smoothness_binary_mask.size())).cuda()
        if raw_mask is not None:
            # everywhere we have non-standard smoothness, slightly reduce
            # the smoothness penalty
            smoothness_weights[raw_mask.data > 0] = args.lambda2
            # on top of cracks and folds only, significantly reduce
            # the smoothness penalty
            smoothness_weights[raw_mask.data > 1] = args.lambda3
        smoothness_weights = masks.contract(
            smoothness_weights, args.mask_smooth_radius,
            ceil=False, binary=False)
        smoothness_weights *= smoothness_binary_mask.float().detach()
        smoothness_weights = F.avg_pool2d(
            smoothness_weights, 2*args.mask_smooth_radius+1, stride=1,
            padding=args.mask_smooth_radius
        )**.5
        if raw_mask is not None:
            # reset the most significant smoothness penalty relaxation
            smoothness_weights[raw_mask.data > 1] = args.lambda3
        smoothness_weights = F.avg_pool2d(smoothness_weights, 5,
                                          stride=1, padding=2)
        if (torch.sum(smoothness_weights[smoothness_binary_mask.byte().data])
                .data[0] < args.eps):
            print('Skipping all zero smoothness weights (factor == 0).')
            visualize_outputs(prefix('zero_smoothness_weights') + '{}',
                              {'src': input_src, 'target': input_target})
            return None

        # smoothness_mask_factor = (dim**2. / torch.sum(
        #     smoothness_weights[smoothness_binary_mask.byte().data]).data[0])
        smoothness_mask_factor = 1

        smoothness_weights /= smoothness_mask_factor
        smoothness_weights = F.grid_sample(smoothness_weights.detach(), field)
        if args.hm:
            smoothness_weights = smoothness_weights.detach()
        smoothness_weights = smoothness_weights * border_mask.float().detach()

        rfield = field - model.pyramid.get_identity_grid(field.size()[-2])
        weighted_smoothness_error_field = smoothness(
            [rfield], weights=smoothness_weights)

        if mask is not None:
            hpred = pred.clone().detach()
            hpred[(mask > 1).detach()] = torch.min(pred).data[0]
        else:
            hpred = pred

        return {
            'src': src,
            'target': target,
            'input_src': input_src.unsqueeze(0).unsqueeze(0),
            'input_target': input_target.unsqueeze(0).unsqueeze(0),
            'field': field,
            'rfield': rfield,
            'residuals': residuals,
            'pred': pred,
            'hpred': hpred,
            'src_mask': mask,
            'raw_src_mask': raw_mask,
            'target_mask': target_mask,
            'similarity_error':
                torch.mean(weighted_similarity_error_field[border_mask]),
            'smoothness_error':
                torch.sum(weighted_smoothness_error_field),
            'smoothness_weights': smoothness_weights,
            'similarity_weights': similarity_weights,
            'similarity_error_field': weighted_similarity_error_field,
            'smoothness_error_field': weighted_smoothness_error_field
        }

    print('=========== BEGIN TRAIN LOOP ============')
    for epoch in range(args.epoch, args.num_epochs):
        print('Beginning training epoch: {}'.format(epoch))
        for t, tensor_dict in enumerate(train_loader):
            if t == 0:
                if epoch % fall_time == 0 and (trunclayer > 0
                                               or args.trunc == 0):
                    # only fine tune if running a tuning session
                    fine_tuning = False or args.fine_tuning
                    if epoch > 0 and trunclayer > 0:
                        trunclayer -= 1
                    optimizer = opt(trunclayer)
                elif (epoch >= fall_time * size - 1
                      or epoch % fall_time == fall_time - 1):
                    if not fine_tuning:
                        fine_tuning = True
                        optimizer = opt(trunclayer)

            # Get inputs
            X = Variable(tensor_dict['X'], requires_grad=False).cuda()
            this_mip = tensor_dict['mip'][0]
            mask_stack = (
                lm_defect_detector.masks_from_stack(X) if this_mip == 2
                else hm_defect_detector.masks_from_stack(X))
            stacks, top, left = aug_stacks(
                [X, mask_stack], padding=padding, jitter=not args.no_jitter,
                jitter_displacement=2**(args.size-1))
            X, mask_stack = stacks[0], stacks[1]

            errs = []
            penalties = []
            consensus_list = []
            contrast_errors = []
            smooth_factor = 1 if trunclayer == 0 and fine_tuning else 0.05
            for sample_idx, i in enumerate(range(1, X.size(1)-1)):
                ##################################
                # RUN SINGLE PAIR OF SLICES ######
                ##################################
                src, target = X[0, i], X[0, i+1]
                src_mask, target_mask = (
                    (mask_stack[0, i], mask_stack[0, i+1]) if trunclayer == 0
                    else (None, None))

                if (min(torch.var(src).data[0], torch.var(target).data[0])
                        < args.blank_var_threshold):
                    print("Skipping blank sections: ({}, {})."
                          .format(torch.var(src).data[0],
                                  torch.var(target).data[0]))
                    visualize_outputs(prefix('blank_sections') + '{}',
                                      {'src': src, 'target': target})
                    continue

                rf, rb = run_pair(src, target, src_mask, target_mask)
                if rf is not None:
                    if not args.pe_only:
                        errs.append(rf['similarity_error'].data[0])
                        penalties.append(rf['smoothness_error'].data[0])
                        if args.lambda6 > 0 and 'consensus' in rf:
                            consensus_list.append(rf['consensus'].data[0])
                    else:
                        contrast_errors.append(rf['contrast_error'].data[0])

                if rb is not None:
                    if not args.pe_only:
                        errs.append(rb['similarity_error'].data[0])
                        penalties.append(rb['smoothness_error'].data[0])
                        if args.lambda6 > 0 and 'consensus' in rb:
                            consensus_list.append(rb['consensus'].data[0])
                    else:
                        contrast_errors.append(rb['contrast_error'].data[0])

                if (sample_idx == 0 and t % args.vis_interval == 0
                        and not args.pe_only):
                    visualize_outputs(prefix('forward') + '{}', rf)
                    visualize_outputs(prefix('backward') + '{}', rb)
                ##################################

                optimizer.step()
                model.zero_grad()

            if args.pe_only:
                mean_contrast = (
                    (sum(contrast_errors) / len(contrast_errors))
                    if len(contrast_errors) > 0 else 0)
                print('Mean contraster error: {}'.format(mean_contrast))
                torch.save(model.state_dict(), 'pt/' + name + '.pt')
            # Save some info
            if len(errs) > 0:
                mean_err_train = sum(errs) / len(errs)
                mean_penalty_train = sum(penalties) / len(penalties)
                mean_consensus = (
                    (sum(consensus_list) / len(consensus_list))
                    if len(consensus_list) > 0 else 0)
                print(t, smooth_factor, trunclayer,
                      mean_err_train + args.lambda1
                      * mean_penalty_train * smooth_factor,
                      mean_err_train, mean_penalty_train, mean_consensus)
                history.append((
                    time.time() - start_time,
                    mean_err_train + mean_penalty_train * smooth_factor,
                    mean_err_train, mean_penalty_train, mean_consensus))
                torch.save(model.state_dict(), 'pt/' + name + '.pt')

                print('Writing status to: {}'.format(log_file))
                with open(log_file, 'a') as log:
                    for tr in history:
                        for val in tr:
                            log.write(str(val) + ', ')
                        log.write('\n')
                    history = []
            else:
                print(
                    "Skipping writing status for stack with no valid slices.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument(
        '--dry_run',
        help='tests data loading when passed, skipping training',
        action='store_true')
    parser.add_argument(
        '--vis_interval',
        help='the number of stacks in between each visualization',
        type=int, default=10)
    parser.add_argument(
        '--mask_smooth_radius',
        help='radius of average pooling kernel for smoothing smoothness '
        'penalty weights', type=int, default=20)
    parser.add_argument(
        '--mask_neighborhood_radius',
        help='radius (in pixels) of neighborhood effects of cracks and folds',
        type=int, default=20)
    parser.add_argument(
        '--eps',
        help='epsilon value to avoid divide-by-zero',
        type=float, default=1e-6)
    parser.add_argument(
        '--no_jitter',
        help='omit jitter when augmenting image stacks', action='store_true')
    parser.add_argument(
        '--hm',
        help='do high mip (mip 5) training run; runs at mip 2 if flag is '
        'omitted', action='store_true')
    parser.add_argument(
        '--mm',
        help='mix mips while training (2,5,7,8)', action='store_true')
    parser.add_argument(
        '--blank_var_threshold',
        help='variance threshold under which an image will be considered '
        'blank and omitted', type=float, default=0.001)
    parser.add_argument(
        '--lambda1', help='total smoothness penalty coefficient',
        type=float, default=0.1)
    parser.add_argument(
        '--lambda2', help='smoothness penalty reduction around cracks/folds',
        type=float, default=0.3)
    parser.add_argument(
        '--lambda3',
        help='smoothness penalty reduction on top of cracks/folds',
        type=float, default=0.00001)
    parser.add_argument(
        '--lambda4', help='MSE multiplier in regions around cracks and folds',
        type=float, default=10)
    parser.add_argument(
        '--lambda5',
        help='MSE multiplier in regions on top of cracks and folds',
        type=float, default=0)
    parser.add_argument(
        '--lambda6',
        help='coefficient for drift-prevention consensus loss in training '
        '(default=0)', type=float, default=0)
    parser.add_argument(
        '--skip',
        help='number of residuals (starting at the bottom of the pyramid) to '
        'omit from the aggregate field', type=int, default=0)
    parser.add_argument(
        '--size',
        help='height of pyramid/number of residual modules in the pyramid',
        type=int, default=5)
    parser.add_argument(
        '--dim',
        help='side dimension of training images', type=int, default=1152)
    parser.add_argument(
        '--trunc',
        help='truncation layer; should usually be size-1 (0 IF FINE TUNING)',
        type=int, default=4)
    parser.add_argument(
        '--lr',
        help='starting learning rate', type=float, default=0.0002)
    parser.add_argument(
        '--num_epochs',
        help='number of training epochs', type=int, default=1000)
    parser.add_argument(
        '--state_archive',
        help='saved model to initialize with', type=str, default=None)
    # parser.add_argument(
    #     '--batch_size',
    #     help='size of batch', type=int, default=1)
    parser.add_argument(
        '--k',
        help='kernel size', type=int, default=7)
    parser.add_argument(
        '--fall_time',
        help='epochs between layers', type=int, default=2)
    parser.add_argument(
        '--epoch',
        help='training epoch to start from', type=int, default=0)
    parser.add_argument(
        '--padding',
        help='amount of padding to add to training stacks',
        type=int, default=128)
    parser.add_argument(
        '--fine_tuning',
        help='when passed, begin with fine tuning (reduce learning rate, '
        'train all parameters)', action='store_true')
    parser.add_argument(
        '--skip_sample_aug',
        help='skips slice-wise augmentation when passed (no cutouts, etc)',
        action='store_true')
    parser.add_argument(
        '--penalty',
        help='type of smoothness penalty (lap, jacob, cjacob, tv)',
        type=str, default='jacob')
    parser.add_argument(
        '--lm_defect_net',
        help='mip2 defect net archive', type=str,
        default='basil_defect_unet18070201')
    parser.add_argument(
        '--hm_defect_net',
        help='mip5 defect net archive', type=str,
        default='basil_defect_unet_mip518070305')
    parser.add_argument(
        '--fine_tuning_lr_factor',
        help='factor by which to reduce learning rate during fine tuning',
        type=float, default=0.2)
    parser.add_argument(
        '--pe_only', action='store_true')
    parser.add_argument(
        '--enc_only', action='store_true')
    parser.add_argument(
        '--vhm_src',
        help='very high mip source (mip 8)', type=str,
        default='~/matriarch_mixed.h5')
    parser.add_argument(
        '--hm_src',
        help='high mip source (mip 5)', type=str, default='~/mip5_mixed.h5')
    parser.add_argument(
        '--lm_src',
        help='low mip source (mip 2)', type=str, default='~/mip2_mixed.h5')
    return parser.parse_args()


if __name__ == '__main__':
    main()
