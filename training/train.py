import os
import sys
import time
import argparse
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
from helpers import gif, save_chunk, center, display_v, dvl, copy_state_to_model, reverse_dim, dilate_mask, contract_mask, invert_mask, union_masks, intersection_masks
from aug import aug_stacks, aug_input, rotate_and_scale, crack, displace_slice
from vis import visualize_outputs
from loss import similarity_score, smoothness_penalty

if __name__ == '__main__':
    identities = {}
    def get_identity_grid(dim):
        if dim not in identities:
            gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
            I = np.stack(np.meshgrid(gx, gy))
            I = np.expand_dims(I, 0)
            I = torch.FloatTensor(I)
            I = torch.autograd.Variable(I, requires_grad=False)
            I = I.permute(0,2,3,1).cuda()
            identities[dim] = I
        return identities[dim]
 
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--eps', help='epsilon value to avoid divide-by-zero', type=float, default=1e-6)
    parser.add_argument('--no_jitter', action='store_true')
    parser.add_argument('--hm', action='store_true')
    parser.add_argument('--unflow', type=float, default=0)
    parser.add_argument('--blank_var_threshold', type=float, default=0.001)
    parser.add_argument('--lambda1', help='total smoothness penalty coefficient', type=float, default=0.1)
    parser.add_argument('--lambda2', help='smoothness penalty reduction around cracks/folds', type=float, default=0.3)
    parser.add_argument('--lambda3', help='smoothness penalty reduction on top of cracks/folds', type=float, default=0.00001)
    parser.add_argument('--lambda4', help='MSE multiplier in regions around cracks and folds', type=float, default=10)
    parser.add_argument('--lambda5', help='MSE multiplier in regions on top of cracks and folds', type=float, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--no_anneal', action='store_true')
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--dim', type=int, default=1152)
    parser.add_argument('--trunc', type=int, default=4)
    parser.add_argument('--lr', help='starting learning rate', type=float, default=0.0002)
    parser.add_argument('--it', help='number of training epochs', type=int, default=1000)
    parser.add_argument('--state_archive', help='saved model to initialize with', type=str, default=None)
    parser.add_argument('--archive_fields', help='whether or not to include residual fields in output', action='store_true')
    parser.add_argument('--batch_size', help='size of batch', type=int, default=1)
    parser.add_argument('--k', help='kernel size', type=int, default=7)
    parser.add_argument('--fall_time', help='epochs between layers', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--padding', type=int, default=128)
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument('--penalty', type=str, default='jacob')
    parser.add_argument('--crack_mask', action='store_false')
    parser.add_argument('--pred', action='store_true')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--skip_sample_aug', action='store_true')
    parser.add_argument('--crack_masks', type=str, default='~/../eam6/basil_raw_cropped_crack_train_mip5.h5')
    parser.add_argument('--fold_masks', type=str, default='~/../eam6/basil_raw_cropped_fold_train_mip5.h5')
    parser.add_argument('--lm_crack_masks', type=str, default='~/../eam6/full_father_crack_train_mip5.h5')
    parser.add_argument('--lm_fold_masks', type=str, default='~/../eam6/full_father_fold_train_mip5.h5')
    args = parser.parse_args()

    name = args.name
    trunclayer = args.trunc
    skiplayers = args.skip
    size = args.size
    padding = args.padding
    dim = args.dim + padding
    kernel_size = args.k
    anneal = not args.no_anneal
    log_path = 'out/' + name + '/'
    log_file = log_path + name + '.log'
    it = args.it
    lr = args.lr
    batch_size = args.batch_size
    fall_time = args.fall_time
    fine_tuning = args.fine_tuning
    epoch = args.epoch
    print(args)

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

    defect_net = torch.load('basil_defect_unet_mip518070305').cpu().cuda() if args.hm else torch.load('basil_defect_unet18070201').cpu().cuda()

    for p in defect_net.parameters():
        p.requires_grad = False
        
    if args.hm:
        train_dataset = StackDataset(os.path.expanduser('~/../eam6/basil_raw_cropped_train_mip5.h5'))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)
    else:
        lm_train_dataset1 = StackDataset(os.path.expanduser('~/../eam6/full_father_train_mip2.h5')) # dataset pulled from all of Basil
        #lm_train_dataset2 = StackDataset(os.path.expanduser('~/../eam6/dense_folds_train_mip2.h5')) # dataset focused on extreme folds
        train_dataset = ConcatDataset([lm_train_dataset1])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)

    test_loader = train_loader

    def opt(layer):
        params = []
        if layer >= skiplayers and not fine_tuning:
            print('training only layer ' + str(layer))
            try:
                params.extend(model.pyramid.mlist[layer].parameters())
            except Exception as e:
                print('Training shared G')
                params.extend(model.pyramid.g.parameters())
        else:
            print('training all residual networks')
            params.extend(model.parameters())

        if fine_tuning or epoch < fall_time - 1:
            print('training ep params')
            params.extend(model.pyramid.enclist.parameters())
        else:
            print('freezing ep params')

        lr_ = lr if not fine_tuning else lr * 0.2
        print('Building optimizer for layer', layer, 'fine tuning', fine_tuning, 'lr', lr_)
        return torch.optim.Adam(params, lr=lr_)

    downsample = lambda x: nn.AvgPool2d(2**x,2**x, count_include_pad=False) if x > 0 else (lambda y: y)
    start_time = time.time()
    mse = similarity_score(should_reduce=False)
    penalty = smoothness_penalty('jacob')
    history = []
    print('=========== BEGIN TRAIN LOOP ============')

    def contrast(t, l=145, h=210):
        zeromask = (t == 0)
        t[t < l] = l
        t[t > h] = h
        t *= 255.0 / (h-l+1)
        t = (t - torch.min(t) + 1) / 255.
        t[zeromask] = 0
        return t
    
    def net_preprocess(stack):
        weights = np.array(
            [[1./48, 1./24, 1./48],
             [1./24, 3./4,  1./24],
             [1./48, 1./24, 1./48]]
        )

        kernel = Variable(torch.FloatTensor(weights).expand(10,1,3,3), requires_grad=False).cuda()
        stack = F.conv2d(stack, kernel, padding=1, groups=10) / 255.0

        for idx in range(stack.size(1)):
            stack[:,idx] = stack[:,idx] - torch.mean(stack[:,idx])
            stack[:,idx] = stack[:,idx] / (torch.std(stack[:,idx]) + 1e-6)
        stack = stack.detach()
        stack.volatile = True
        return stack

    def net_postprocess(raw_output, binary_threshold=0.4, minor_dilation_radius=1, major_dilation_radius=75):
        sigmoided = F.sigmoid(raw_output)
        pooled = F.max_pool2d(sigmoided, minor_dilation_radius*2+1, stride=1, padding=minor_dilation_radius)
        smoothed = F.avg_pool2d(pooled, minor_dilation_radius*2+1, stride=1, padding=minor_dilation_radius, count_include_pad=False)
        smoothed[smoothed > binary_threshold] = 1
        smoothed[smoothed <= binary_threshold] = 0
        dilated = F.max_pool2d(smoothed, major_dilation_radius*2+1, stride=1, padding=major_dilation_radius)
        final_output = smoothed + dilated
        final_output.volatile = False
        return final_output

    def cfm_from_stack(stack):
        stack = net_preprocess(stack).detach()
        raw_output = torch.cat([torch.max(defect_net(stack[:,i:i+1]),1,keepdim=True)[0] for i in range(stack.size(1))], 1)
        final_output = net_postprocess(raw_output)
        return final_output
        
    def run_sample(X, mask=None, target_mask=None, train=True, vis=False):
        model.train(train)

        # our convention here is that X is 4D PyTorch convolution shape, while src and target are in squeezed shape (2D)
        src, target = X[0,0], torch.squeeze(X[0,1:])
        
        if train and not args.skip_sample_aug:
            # random rotation
            should_rotate = random.randint(0,1) == 0
            if should_rotate:
                src, grid = rotate_and_scale(src.unsqueeze(0).unsqueeze(0), None)
                target = rotate_and_scale(target.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                if mask is not None:
                    mask = rotate_and_scale(mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                    target_mask = rotate_and_scale(target_mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                    
                src = src.squeeze()
        
        input_src = src.clone()
        input_target = target.clone()

        src_cutout_masks = []
        target_cutout_masks = []

        if train and not args.skip_sample_aug:
            input_src, src_cutout_masks = aug_input(input_src)
            input_target, target_cutout_masks = aug_input(input_target)
        elif train:
            print 'Skipping sample-wise augmentation...'

        pred, field, residuals = model.apply(input_src, input_target, trunclayer)

        # resample tensors that are in our source coordinate space with our new field prediction
        # to move them into target coordinate space so we can compare things fairly
        pred = F.grid_sample(downsample(trunclayer)(src.unsqueeze(0).unsqueeze(0)), field)
        if mask is not None:
            mask = torch.ceil(F.grid_sample(downsample(trunclayer)(mask.unsqueeze(0).unsqueeze(0)), field))
        if len(src_cutout_masks) > 0:
            src_cutout_masks = [torch.ceil(F.grid_sample(m.float(), field)).byte() for m in src_cutout_masks]
        
        target = downsample(trunclayer)((target).unsqueeze(0).unsqueeze(0))

        # mask away similarity error from pixels outside the boundary of either prediction or target
        # in other words, we need valid data in both the target and prediction to count the error from
        # that pixel
        border_mse_mask = (pred != 0) * (target != 0)
        border_mse_mask, no_valid_pixels = contract_mask(border_mse_mask, 5)
        border_mse_mask = border_mse_mask.float()
        if no_valid_pixels:
            print('Empty mask, skipping!')
            return None

        # mask away similarity error from pixels within cutouts in either the source or target
        cutout_mse_masks = src_cutout_masks + target_cutout_masks
        if len(cutout_mse_masks) > 0:
            cutout_mse_mask = union_masks(cutout_mse_masks)
            cutout_mse_mask = invert_mask(cutout_mse_mask).float()
        else:
            cutout_mse_mask = border_mse_mask

        # mask away similarity error for pixels within a defect in the target image
        if target_mask is not None:
            target_defect_mse_mask = (target_mask < 2).float().detach()
        else:
            target_defect_mse_mask = border_mse_mask
            
        mse_weights = border_mse_mask * cutout_mse_mask * target_defect_mse_mask

        # reweight for focus areas
        if mask is not None:
            mse_weights.data[mask.data == 1] = mse_weights.data[mask.data == 1] * args.lambda4
            mse_weights.data[mask.data > 1] = mse_weights.data[mask.data > 1] * args.lambda5

        err = mse(pred, target)
        mse_mask_factor = (torch.sum(mse_weights[border_mse_mask.byte().detach()]) / torch.sum(border_mse_mask)).data[0]
        if mse_mask_factor < args.eps:
            print('Similarity mask factor zero; skipping!')
            return None

        merr = err * (mse_weights / mse_mask_factor)

        smoothness_mask = torch.max(mask, 2 * (pred == 0).float()) if mask is not None else 2 * (pred == 0).float()

        smoothness_weights = Variable(torch.ones(smoothness_mask.size()).cuda())
        smoothness_weights[(smoothness_mask > 0).data] = args.lambda2 # everywhere we have non-standard smoothness, slightly reduce the smoothness penalty
        smoothness_weights[(smoothness_mask > 1).data] = args.lambda3 # on top of cracks and folds only, significantly reduce the smoothness penalty
        smoothness_mask_factor = (torch.sum(smoothness_weights[border_mse_mask.byte().detach()]) / torch.sum(border_mse_mask)).data[0]
        if smoothness_mask_factor < args.eps:
            print('Smoothness mask factor zero; skipping!')
            return None

        smoothness_weights = smoothness_weights / smoothness_mask_factor

        print mse_mask_factor, smoothness_mask_factor
        
        if mask is not None:
            hpred = pred.clone().detach()
            hpred[(mask > 1).detach()] = torch.min(pred).data[0]
        else:
            hpred = pred

        return {
            'input_src' : input_src,
            'input_target' : input_target,
            'pred' : pred,
            'field' : field,
            'rfield' : field - get_identity_grid(field.size()[-2]),
            'residuals' : residuals,
            'similarity_error' : torch.mean(merr),
            'smoothness_weights' : smoothness_weights,
            'similarity_weights' : mse_weights,
            'hpred' : hpred,
            'src_mask' : mask,
            'target_mask' : target_mask,
            'similarity_error_field' : merr
        }

    X_test = None
    for idxx, tensor_dict in enumerate(test_loader):
        X_test = Variable(tensor_dict['X']).cuda().detach()
        X_test.volatile = True
        if idxx > 1:
            break

    for epoch in range(args.epoch, it):
        print('epoch', epoch)
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
            mask_stack = cfm_from_stack(X)
            stacks, top, left = aug_stacks([contrast(X), mask_stack], padding=padding, jitter=not args.no_jitter)
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
                src_mask, target_mask = mask_stack[0,i], mask_stack[0,i+1]

                if min(torch.var(X_[0,0]).data[0], torch.var(X_[0,1]).data[0]) < args.blank_var_threshold:
                    print "Skipping blank sections", torch.var(X_[0,0]).data[0], torch.var(X_[0,1]).data[0]
                    continue

                rf = run_sample(X_, src_mask, target_mask, train=True)
                if rf is not None:
                    smoothness_error_field = args.lambda1 * penalty([rf['rfield']], weights=rf['smoothness_weights'])
                    rf['smoothness_error_field'] = smoothness_error_field
                    cost = rf['similarity_error'] + smooth_factor * torch.sum(smoothness_error_field)
                    (cost/2).backward(retain_graph=args.unflow>0)
                    errs.append(rf['similarity_error'].data[0])
                    penalties.append(torch.sum(smoothness_error_field).data[0])

                ##################################
                # RUN SAMPLE BACKWARD ############
                ##################################
                X_ = reverse_dim(X_,1)
                src_mask, target_mask = target_mask, src_mask

                rb = run_sample(X_, src_mask, target_mask, train=True)
                if rb is not None:
                    smoothness_error_field = args.lambda1 * penalty([rb['rfield']], weights=rb['smoothness_weights'])
                    rb['smoothness_error_field'] = smoothness_error_field
                    cost = rb['similarity_error'] + smooth_factor * torch.sum(smoothness_error_field)
                    (cost/2).backward(retain_graph=args.unflow>0)
                    errs.append(rb['similarity_error'].data[0])
                    penalties.append(torch.sum(smoothness_error_field).data[0])

                ##################################
                # VISUALIZATION ##################
                ##################################
                if sample_idx == 0 and t % 6 == 0:
                    prefix = '{}{}_e{}_t{}'.format(log_path, name, epoch, t)
                    visualize_outputs(prefix + '_forward_{}', rf)
                    visualize_outputs(prefix + '_backward_{}', rb)
                
                ##################################
                # CONSENSUS PENALTY COMPUTATION ##
                ##################################
                if args.unflow > 0 and rf is not None and rb is not None:
                    print 'Computing consensus'
                    ffield, rffield = rf['field'], rf['rfield']
                    bfield, rbfield = rb['field'], rb['rfield']
                    consensus_forward = torch.mean((F.grid_sample(rbfield.permute(0,3,1,2), ffield).permute(0,2,3,1) + rffield) ** 2)
                    consensus_backward = torch.mean((F.grid_sample(rffield.permute(0,3,1,2), bfield).permute(0,2,3,1) + rbfield) ** 2)
                    consensus_unweighted = consensus_forward + consensus_backward
                    consensus = args.unflow * consensus_unweighted
                    consensus.backward()
                    consensus_list.append(consensus.data[0])
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

                print('Writing status to: ', log_file)
                with open(log_file, 'a') as log:
                    for tr in history:
                        for val in tr:
                            log.write(str(val) + ', ')
                        log.write('\n')
                    history = []
            else:
                print "Skipped writing status for blank stack"
