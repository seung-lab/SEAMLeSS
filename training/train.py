if __name__ == '__main__': 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--dry_run', help='tests data loading when passed, skipping training', action='store_true')
    parser.add_argument('--vis_interval', help='the number of stacks in between each visualization', type=int, default=10)
    parser.add_argument('--mask_smooth_radius', help='radius of average pooling kernel for smoothing smoothness penalty weights', type=int, default=20)
    parser.add_argument('--mask_neighborhood_radius', help='radius (in pixels) of neighborhood effects of cracks and folds', type=int, default=20)
    parser.add_argument('--eps', help='epsilon value to avoid divide-by-zero', type=float, default=1e-6)
    parser.add_argument('--no_jitter', help='omit jitter when augmenting image stacks', action='store_true')
    parser.add_argument('--hm', help='do high mip (mip 5) training run; runs at mip 2 if flag is omitted', action='store_true')
    parser.add_argument('--mm', help='mix mips while training (2,5,7,8)', action='store_true')
    parser.add_argument('--unflow', help='coefficient for \'unflow\' consensus loss in training (default=0)', type=float, default=0)
    parser.add_argument('--blank_var_threshold', help='variance threshold under which an image will be considered blank and omitted', type=float, default=0.001)
    parser.add_argument('--lambda1', help='total smoothness penalty coefficient', type=float, default=0.1)
    parser.add_argument('--lambda2', help='smoothness penalty reduction around cracks/folds', type=float, default=0.3)
    parser.add_argument('--lambda3', help='smoothness penalty reduction on top of cracks/folds', type=float, default=0.00001)
    parser.add_argument('--lambda4', help='MSE multiplier in regions around cracks and folds', type=float, default=10)
    parser.add_argument('--lambda5', help='MSE multiplier in regions on top of cracks and folds', type=float, default=0)
    parser.add_argument('--skip', help='number of residuals (starting at the bottom of the pyramid) to omit from the aggregate field', type=int, default=0)
    parser.add_argument('--no_anneal', help='do not anneal the smoothness penalty in the early stages of training', action='store_true')
    parser.add_argument('--size', help='height of pyramid/number of residual modules in the pyramid', type=int, default=5)
    parser.add_argument('--dim', help='side dimension of training images', type=int, default=1152)
    parser.add_argument('--trunc', help='truncation layer; should usually be size-1 (0 IF FINE TUNING)', type=int, default=4)
    parser.add_argument('--lr', help='starting learning rate', type=float, default=0.0002)
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=1000)
    parser.add_argument('--state_archive', help='saved model to initialize with', type=str, default=None)
    parser.add_argument('--batch_size', help='size of batch', type=int, default=1)
    parser.add_argument('--k', help='kernel size', type=int, default=7)
    parser.add_argument('--fall_time', help='epochs between layers', type=int, default=2)
    parser.add_argument('--epoch', help='training epoch to start from', type=int, default=0)
    parser.add_argument('--padding', help='amount of padding to add to training stacks', type=int, default=128)
    parser.add_argument('--fine_tuning', help='when passed, begin with fine tuning (reduce learning rate, train all parameters)', action='store_true')
    parser.add_argument('--skip_sample_aug', help='skips slice-wise augmentation when passed (no cutouts, etc)', action='store_true')
    parser.add_argument('--penalty', help='type of smoothness penalty (lap, jacob, cjacob, tv)', type=str, default='jacob')
    parser.add_argument('--lm_defect_net', help='mip2 defect net archive', type=str, default='basil_defect_unet18070201')
    parser.add_argument('--hm_defect_net', help='mip5 defect net archive', type=str, default='basil_defect_unet_mip518070305')
    parser.add_argument('--fine_tuning_lr_factor', help='factor by which to reduce learning rate during fine tuning', type=float, default=0.2)
    args = parser.parse_args()

    import os
    import sys
    import time
    import operator
    
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from torch.optim import lr_scheduler
    from torch.autograd import Variable
    import numpy as np
    import collections
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import random
    
    from stack_dataset import StackDataset
    from torch.utils.data import DataLoader, ConcatDataset
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        import h5py
        
    from pyramid import PyramidTransformer
    from defect_net import *
    from defect_detector import DefectDetector
    from helpers import gif, save_chunk, center, display_v, dvl, copy_state_to_model, reverse_dim
    import masks
    from aug import aug_stacks, aug_input, rotate_and_scale, crack, displace_slice
    from vis import visualize_outputs
    from loss import similarity_score, smoothness_penalty
    from normalizer import Normalizer

    name = args.name
    trunclayer = args.trunc
    skiplayers = args.skip
    size = args.size
    fall_time = args.fall_time
    padding = args.padding
    dim = args.dim + padding
    kernel_size = args.k
    anneal = not args.no_anneal
    log_path = 'out/' + name + '/'
    log_file = log_path + name + '.log'
    lr = args.lr
    batch_size = args.batch_size
    fine_tuning = args.fine_tuning
    epoch = args.epoch

    downsample = lambda x: nn.AvgPool2d(2**x,2**x, count_include_pad=False) if x > 0 else (lambda y: y)
    start_time = time.time()
    mse = similarity_score(should_reduce=False)
    penalty = smoothness_penalty(args.penalty)
    history = []

    if args.mm or args.hm:
        hm_defect_detector = DefectDetector(torch.load(args.hm_defect_net).cpu().cuda(),
                                            major_dilation_radius=args.mask_neighborhood_radius,
                                            sigmoid_threshold = 0.5 if args.hm else 0.4)
    if args.mm or (not args.hm):
        lm_defect_detector = DefectDetector(torch.load(args.lm_defect_net).cpu().cuda(),
                                            major_dilation_radius=args.mask_neighborhood_radius,
                                            sigmoid_threshold = 0.5 if args.hm else 0.4)
    normalizer = Normalizer(5 if args.hm else 2)

    if args.mm:
        train_dataset1 = StackDataset(os.path.expanduser('~/../eam6/mip5_mixed.h5'), mip=5)
        train_dataset2 = StackDataset(os.path.expanduser('~/../eam6/mip2_mixed.h5'), mip=2)
        train_dataset3 = StackDataset(os.path.expanduser('~/../eam6/matriarch_mixed.h5'), mip=8)
        train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])
    else:
        if args.hm:
            train_dataset1 = StackDataset(os.path.expanduser('~/../eam6/mip5_mixed.h5'), mip=5)
            train_dataset2 = StackDataset(os.path.expanduser('~/../eam6/matriarch_mixed.h5'), mip=5)
            train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        else:
            #train_dataset1 = StackDataset(os.path.expanduser('~/../eam6/full_father_train_mip2.h5')) # dataset pulled from all of Basil
            train_dataset2 = StackDataset(os.path.expanduser('~/../eam6/mip2_mixed.h5'), mip=2) # dataset focused on extreme folds
            train_dataset = ConcatDataset([train_dataset2])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir('pt'):
        os.makedirs('pt')

    with open(log_path + 'args.txt', 'a') as f:
        f.write(str(args))
    
    if args.state_archive is None:
        model = PyramidTransformer(size=size, dim=dim, skip=skiplayers, k=kernel_size).cuda()
    else:
        model = PyramidTransformer.load(args.state_archive, height=size, dim=dim, skips=skiplayers, k=kernel_size)
        for p in model.parameters():
            p.requires_grad = True
        model.train(True)

    def opt(layer):
        params = []
        if layer >= skiplayers and not fine_tuning:
            print('Training only layer {}'.format(layer))
            params.extend(model.pyramid.mlist[layer].parameters())
        else:
            print('Training all residual networks.')
            params.extend(model.pyramid.mlist.parameters())

        if fine_tuning or epoch < fall_time - 1:
            print('Training all encoder parameters.')
            params.extend(model.pyramid.enclist.parameters())
        else:
            print('Freezing encoder parameters.')

        lr_ = lr if not fine_tuning else lr * args.fine_tuning_lr_factor
        print('Building optimizer for layer {} (fine tuning: {}, lr: {})'.format(layer, fine_tuning, lr_))
        return torch.optim.Adam(params, lr=lr_)

    def run_sample(X, mask=None, target_mask=None, train=True):
        model.train(train)

        # our convention here is that X is 4D PyTorch convolution shape, while src and target are in squeezed shape (2D)
        src, target = X[0,0], torch.squeeze(X[0,1:])
        src = Variable(torch.FloatTensor(normalizer.apply(src.data.cpu().numpy()))).cuda()
        target = Variable(torch.FloatTensor(normalizer.apply(target.data.cpu().numpy()))).cuda()

        if train and not args.skip_sample_aug:
            # random rotation
            should_rotate = random.randint(0,1) == 0
            if should_rotate:
                src, grid = rotate_and_scale(src.unsqueeze(0).unsqueeze(0), None)
                target = rotate_and_scale(target.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                if mask is not None:
                    mask = torch.ceil(rotate_and_scale(mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze())
                    target_mask = torch.ceil(rotate_and_scale(target_mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze())
                    
                src = src.squeeze()
        
        input_src = src.clone()
        input_target = target.clone()

        target = downsample(trunclayer)(target.unsqueeze(0).unsqueeze(0))

        src_cutout_masks = []
        target_cutout_masks = []

        if train and not args.skip_sample_aug:
            input_src, src_cutout_masks = aug_input(input_src)
            input_target, target_cutout_masks = aug_input(input_target)
        elif train:
            print('Skipping sample-wise augmentation...')

        if args.dry_run:
            print('Bailing for dry run [not an error].')
            return None

        pred, field, residuals = model.apply(input_src, input_target, trunclayer)

        # resample tensors that are in our source coordinate space with our new field prediction
        # to move them into target coordinate space so we can compare things fairly
        pred = F.grid_sample(downsample(trunclayer)(src.unsqueeze(0).unsqueeze(0)), field)
        raw_mask = mask
        if mask is not None:
            mask = torch.ceil(F.grid_sample(downsample(trunclayer)(mask.unsqueeze(0).unsqueeze(0)), field))
        if target_mask is not None:
            target_mask = masks.dilate(target_mask.unsqueeze(0).unsqueeze(0), 1, binary=False)
        if len(src_cutout_masks) > 0:
            src_cutout_masks = [torch.ceil(F.grid_sample(m.float(), field)).byte() for m in src_cutout_masks]
        
        # first we'll build a binary mask to completely ignore 'irrelevant' pixels
        mse_binary_masks = []

        # mask away similarity error from pixels outside the boundary of either prediction or target
        # in other words, we need valid data in both the target and prediction to count the error from
        # that pixel
        border_mse_mask = masks.intersect([pred != 0, target != 0])
        border_mse_mask, no_valid_pixels = masks.contract(border_mse_mask, 5, return_sum=True)
        if no_valid_pixels:
            print('Empty mask, skipping!')
            return None
        mse_binary_masks.append(border_mse_mask)
        
        # mask away similarity error from pixels within cutouts in either the source or target
        cutout_mse_masks = src_cutout_masks + target_cutout_masks
        if len(cutout_mse_masks) > 0:
            cutout_mse_mask = masks.union(cutout_mse_masks)
            cutout_mse_mask = masks.invert(cutout_mse_mask)
            mse_binary_masks.append(cutout_mse_mask)

        # mask away similarity error for pixels within a defect in the target image
        if target_mask is not None:
            target_defect_mse_mask = masks.contract(target_mask < 2, 2).detach()
            mse_binary_masks.append(target_defect_mse_mask)
            
        mse_binary_mask = masks.intersect(mse_binary_masks)

        # reweight remaining pixels for focus areas
        mse_weights = Variable(torch.ones(border_mse_mask.size())).cuda()
        if mask is not None:
            mse_weights.data[masks.intersect([mask > 0, mask <= 1]).data] = args.lambda4
            mse_weights.data[masks.dilate(mask > 1, 2).data] = args.lambda5
        #if target_mask is not None:
        #    mse_weights.data[masks.intersect([target_mask.data > 0, target_mask.data <= 1, mask.data <= 1])] *= args.lambda4

        mse_weights *= mse_binary_mask.float()        
        mse_mask_factor = (torch.sum(mse_weights[border_mse_mask.detach()]) / torch.sum(border_mse_mask.float())).data[0]
        if mse_mask_factor < args.eps:
            print('Similarity mask factor zero; skipping')
            return None

        # compute raw similarity error and masked/weighted similarity error
        similarity_error_field = mse(pred, target)
        weighted_similarity_error_field = similarity_error_field * (mse_weights / mse_mask_factor)

        # perform masking/weighting for smoothness penalty
        smoothness_binary_mask = masks.intersect([src != 0, target != 0]).view(mse_binary_mask.size())
        smoothness_weights = Variable(torch.ones(smoothness_binary_mask.size())).cuda()
        if raw_mask is not None:
            smoothness_weights[raw_mask.data > 0] = args.lambda2 # everywhere we have non-standard smoothness, slightly reduce the smoothness penalty
            smoothness_weights[raw_mask.data > 1] = args.lambda3 # on top of cracks and folds only, significantly reduce the smoothness penalty
        smoothness_weights = masks.contract(smoothness_weights, args.mask_smooth_radius, ceil=False, binary=False)
        smoothness_weights *= smoothness_binary_mask.float().detach()
        smoothness_weights = F.avg_pool2d(smoothness_weights, 2*args.mask_smooth_radius+1, stride=1, padding=args.mask_smooth_radius)**.5
        if raw_mask is not None:
            smoothness_weights[raw_mask.data > 1] = args.lambda3 # reset the most significant smoothness penalty relaxation
        smoothness_weights = F.avg_pool2d(smoothness_weights, 5, stride=1, padding=2)
        smoothness_mask_factor = (torch.sum(smoothness_weights[smoothness_binary_mask.byte().detach()]) / torch.sum(smoothness_binary_mask.float())).data[0]
        if smoothness_mask_factor < args.eps:
            print('Smoothness mask factor zero; skipping')
            return None

        smoothness_weights /= smoothness_mask_factor
        smoothness_weights = F.grid_sample(smoothness_weights.detach(), field)
        if args.hm:
            smoothness_weights = smoothness_weights.detach()
        smoothness_weights = smoothness_weights * border_mse_mask.float().detach()
        
        rfield = field - model.pyramid.get_identity_grid(field.size()[-2])
        smoothness_error_field = penalty([rfield], weights=smoothness_weights)
        
        if mask is not None:
            hpred = pred.clone().detach()
            hpred[(mask > 1).detach()] = torch.min(pred).data[0]
        else:
            hpred = pred

        return {
            'input_src' : input_src,
            'input_target' : input_target,
            'field' : field,
            'rfield' : rfield,
            'residuals' : residuals,
            'pred' : pred,
            'hpred' : hpred,
            'src_mask' : mask,
            'raw_src_mask' : raw_mask,
            'target_mask' : target_mask,
            'similarity_error' : torch.mean(weighted_similarity_error_field),
            'smoothness_error' : torch.sum(smoothness_error_field),
            'smoothness_weights' : smoothness_weights,
            'similarity_weights' : mse_weights,
            'similarity_error_field' : weighted_similarity_error_field,
            'smoothness_error_field' : smoothness_error_field
        }

    print('=========== BEGIN TRAIN LOOP ============')
    for epoch in range(args.epoch, args.num_epochs):
        print('Beginning training epoch: {}'.format(epoch))
        for t, tensor_dict in enumerate(train_loader):
            if t == 0:
                if epoch % fall_time == 0 and (trunclayer > 0 or args.trunc == 0):
                    fine_tuning = False or args.fine_tuning # only fine tune if running a tuning session
                    if epoch > 0 and trunclayer > 0:
                        trunclayer -= 1
                    optimizer = opt(trunclayer)
                elif epoch >= fall_time * size - 1 or epoch % fall_time == fall_time - 1:
                    if not fine_tuning:
                        fine_tuning = True
                        optimizer = opt(trunclayer)

            # Get inputs
            X = Variable(tensor_dict['X'], requires_grad=False).cuda()
            this_mip = tensor_dict['mip'][0]
            mask_stack = lm_defect_detector.masks_from_stack(X) if this_mip == 2 else hm_defect_detector.masks_from_stack(X)
            #X = Variable(torch.FloatTensor(normalizer.apply(X.data.cpu().numpy()))).cuda()
            stacks, top, left = aug_stacks([X, mask_stack], padding=padding, jitter=not args.no_jitter, jitter_displacement=2**(args.size-1))
            X, mask_stack = stacks[0], stacks[1]

            errs = []
            penalties = []
            consensus_list = []
            smooth_factor = 1 if trunclayer == 0 and fine_tuning else 0.05
            for sample_idx, i in enumerate(range(1,X.size(1)-1)):
                ##################################
                # RUN SAMPLE FORWARD #############
                ##################################
                X_ = X[:,i:i+2].detach()
                src_mask, target_mask = (mask_stack[0,i], mask_stack[0,i+1]) if trunclayer == 0 else (None, None)

                if min(torch.var(X_[0,0]).data[0], torch.var(X_[0,1]).data[0]) < args.blank_var_threshold:
                    print("Skipping blank sections: ({}, {})".format(torch.var(X_[0,0]).data[0], torch.var(X_[0,1]).data[0]))
                    continue

                rf = run_sample(X_, src_mask, target_mask, train=True)
                if rf is not None:
                    cost = rf['similarity_error'] + args.lambda1 * smooth_factor * rf['smoothness_error']
                    (cost/2).backward(retain_graph=args.unflow>0)
                    errs.append(rf['similarity_error'].data[0])
                    penalties.append(rf['smoothness_error'].data[0])

                ##################################
                # RUN SAMPLE BACKWARD ############
                ##################################
                X_ = reverse_dim(X_,1)
                src_mask, target_mask = target_mask, src_mask

                rb = run_sample(X_, src_mask, target_mask, train=True)
                if rb is not None:
                    cost = rb['similarity_error'] + args.lambda1 * smooth_factor * rb['smoothness_error']
                    (cost/2).backward(retain_graph=args.unflow>0)
                    errs.append(rb['similarity_error'].data[0])
                    penalties.append(rb['smoothness_error'].data[0])

                ##################################
                # CONSENSUS PENALTY COMPUTATION ##
                ##################################
                if args.unflow > 0 and rf is not None and rb is not None:
                    ffield, rffield = rf['field'], rf['rfield']
                    bfield, rbfield = rb['field'], rb['rfield']
                    consensus_forward = (F.grid_sample(rbfield.permute(0,3,1,2), ffield).permute(0,2,3,1) + rffield) ** 2
                    consensus_backward = (F.grid_sample(rffield.permute(0,3,1,2), bfield).permute(0,2,3,1) + rbfield) ** 2
                    rf['consensus'] = consensus_forward
                    rb['consensus'] = consensus_backward
                    consensus_unweighted = torch.mean(consensus_forward) + torch.mean(consensus_backward)
                    consensus = args.unflow * consensus_unweighted
                    consensus.backward()
                    consensus_list.append(consensus.data[0])

                ##################################
                # VISUALIZATION ##################
                ##################################
                if sample_idx == 0 and t % args.vis_interval == 0:
                    prefix = '{}{}_e{}_t{}'.format(log_path, name, epoch, t)
                    visualize_outputs(prefix + '_forward_{}', rf)
                    visualize_outputs(prefix + '_backward_{}', rb)
                ##################################

                optimizer.step()
                model.zero_grad()

            # Save some info
            if len(errs) > 0:
                mean_err_train = sum(errs) / len(errs)
                mean_penalty_train = sum(penalties) / len(penalties)
                mean_consensus = (sum(consensus_list) / len(consensus_list)) if len(consensus_list) > 0 else 0
                print(t, smooth_factor, trunclayer, mean_err_train + mean_penalty_train * smooth_factor, mean_err_train, mean_penalty_train, mean_consensus)
                history.append((time.time() - start_time, mean_err_train + mean_penalty_train * smooth_factor, mean_err_train, mean_penalty_train, mean_consensus))
                torch.save(model.state_dict(), 'pt/' + name + '.pt')

                print('Writing status to: {}'.format(log_file))
                with open(log_file, 'a') as log:
                    for tr in history:
                        for val in tr:
                            log.write(str(val) + ', ')
                        log.write('\n')
                    history = []
            else:
                print("Skipped writing status for blank stack...")
