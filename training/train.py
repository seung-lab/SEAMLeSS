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
from dnet import UNet
from helpers import gif, save_chunk, center, display_v, dvl, copy_state_to_model, reverse_dim
from aug import aug_stacks, aug_input, rotate_and_scale, crack, displace_slice

from loss import similarity_score, smoothness_penalty
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

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
    parser.add_argument('--target_weights', type=arg_as_list, default=None)
    parser.add_argument('--no_jitter', action='store_true')
    parser.add_argument('--hm', action='store_true')
    parser.add_argument('--unflow', type=float, default=0)
    parser.add_argument('--alpha', help='(1 - proportion) of prediction to use for alignment when aligning to predictions', type=float, default=1)
    parser.add_argument('--blank_var_threshold', type=float, default=0.001)
    parser.add_argument('--num_targets', type=int, default=1)
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
    parser.add_argument('--inference_only', help='whether or not to skip training', action='store_true')
    parser.add_argument('--archive_fields', help='whether or not to include residual fields in output', action='store_true')
    parser.add_argument('--batch_size', help='size of batch', type=int, default=1)
    parser.add_argument('--k', help='kernel size', type=int, default=7)
    parser.add_argument('--dilate', help='use dilation for G_i', action='store_true')
    parser.add_argument('--amp', help='amplify G_i', action='store_true')
    parser.add_argument('--unet', help='use unet for G_i', action='store_true')
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
    amp = args.amp
    unet = args.unet
    dilate = args.dilate
    trunclayer = args.trunc
    skiplayers = args.skip
    num_targets = args.num_targets
    size = args.size
    padding = args.padding
    dim = args.dim + padding
    kernel_size = args.k
    lambda1 = args.lambda1
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

    if args.inference_only:
        trunclayer = 0

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir('pt'):
        os.makedirs('pt')

    with open(log_path + 'args.txt', 'a') as f:
        f.write(str(args))
    
    if args.pred:
        print('Loading prednet...')
        prednet = UNet(3,1).cuda()
        prednet.load_state_dict(torch.load('../framenet/pt/dnet.pt'))
        
    if args.state_archive is None:
        model = PyramidTransformer(size=size, dim=dim, skip=skiplayers, k=kernel_size, dilate=dilate, amp=amp, unet=unet, num_targets=num_targets, name=log_path + name, target_weights=(tuple(args.target_weights) if args.target_weights is not None else None)).cuda()
    else:
        model = PyramidTransformer.load(args.state_archive, height=size, dim=dim, skips=skiplayers, k=kernel_size, dilate=dilate, unet=unet, num_targets=num_targets, name=log_path + name, target_weights=(tuple(args.target_weights) if args.target_weights is not None else None))

    for p in model.parameters():
        p.requires_grad = not args.inference_only
    model.train(not args.inference_only)

    if args.hm:
        train_dataset = StackDataset(os.path.expanduser('~/../eam6/basil_raw_cropped_train_mip5.h5'), os.path.expanduser(args.crack_masks) if args.crack_masks is not None else None, os.path.expanduser(args.fold_masks) if args.fold_masks is not None else None, basil=True, threshold_masks=True, combine_masks=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)
    else:
        lm_train_dataset1 = StackDataset(os.path.expanduser('~/../eam6/full_father_train_mip2.h5'), os.path.expanduser(args.lm_crack_masks) if args.lm_crack_masks is not None else None, os.path.expanduser(args.lm_fold_masks) if args.lm_fold_masks is not None else None, basil=True, threshold_masks=True, combine_masks=True, lm=True) # dataset pulled from all of Basil
        lm_train_dataset2 = StackDataset(os.path.expanduser('~/../eam6/dense_folds_train_mip2.h5'), os.path.expanduser('~/../eam6/dense_folds_crack_train_mip5.h5'), os.path.expanduser('~/../eam6/dense_folds_fold_train_mip5.h5'), basil=True, threshold_masks=True, combine_masks=True, lm=True) # dataset focused on extreme folds
        train_dataset = ConcatDataset([lm_train_dataset1, lm_train_dataset2])
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

    def run_sample(X, mask=None, target_mask=None, train=True, vis=False):
        model.train(train)
        src, target = X[0,0], torch.squeeze(X[0,1:])
        
        if train and not args.skip_sample_aug:
            # random flip
            should_rotate = random.randint(0,1) == 0
            if should_rotate:
                src, grid = rotate_and_scale(src.unsqueeze(0).unsqueeze(0), None)
                target = rotate_and_scale(target.unsqueeze(0).unsqueeze(0) if num_targets == 1 else target.unsqueeze(0), grid=grid)[0].squeeze()
                if mask is not None:
                    mask = rotate_and_scale(mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                    target_mask = rotate_and_scale(target_mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                    
                src = src.squeeze()
        
        input_src = src.clone()
        input_target = target.clone()

        src_cutout_masks = []
        target_cutout_masks = []

        if train and not args.skip_sample_aug:
            if random.randint(0,1) == 0:
                input_src, src_cutout_masks = aug_input(input_src)
            if random.randint(0,1) == 0:
                if num_targets > 1:
                    good_slice = random.randint(0, num_targets - 1)
                    black_slice = random.randint(0, num_targets - 1)
                    input_targets, target_cutout_mask_arrays = [], []
                    for target_idx in range(num_targets):
                        if target_idx != good_slice:
                            aug_target, target_cutout_masks = aug_input(input_target[target_idx].clone())
                            input_targets.append(aug_target)
                            target_cutout_mask_arrays.append(target_cutout_masks)
                        else:
                            input_targets.append(input_target[target_idx])
                            target_cutout_masks.append([])
                        if target_idx == black_slice and random.randint(0,1) == 0:
                            input_targets[target_idx] = Variable(torch.zeros(input_targets[target_idx].size())).cuda()
                    target_cutout_masks = [inner for outer in target_cutout_mask_arrays for inner in outer]
                    target_cutout_masks = []
                    input_target = torch.cat([s.unsqueeze(0) for s in input_targets], 0)
                else:
                    input_target, target_cutout_masks = aug_input(input_target)
        else:
            print 'Skipping sample-wise augmentation...'
        pred, field, residuals = model.apply(input_src, input_target, trunclayer)

        # resample with our new field prediction
        pred = F.grid_sample(downsample(trunclayer)(src.unsqueeze(0).unsqueeze(0)), field)
        if mask is not None:
            mask = F.grid_sample(downsample(trunclayer)(mask.unsqueeze(0).unsqueeze(0)), field)
        if len(src_cutout_masks) > 0:
            src_cutout_masks = [F.grid_sample(m.float(), field).byte() for m in src_cutout_masks]
        
        rfield = field - get_identity_grid(field.size()[-2])
        target = downsample(trunclayer)((target[0] if num_targets > 1 else target).unsqueeze(0).unsqueeze(0))

        border_mse_mask = -nn.MaxPool2d(11,1,5)(-Variable((pred.data != 0) * (target.data != 0)).float())
        crack_fold_mse_mask = Variable(torch.ones(border_mse_mask.size())).cuda()
        if target_mask is not None:
            crack_fold_mse_mask[target_mask > 1] = 0
        cutout_mse_masks = src_cutout_masks + target_cutout_masks
        if len(cutout_mse_masks) > 0:
            cutout_mse_mask = torch.sum(torch.cat(cutout_mse_masks,0),0)
            cutout_mse_mask[(cutout_mse_mask>0).detach()] = 1
            cutout_mse_mask = nn.MaxPool2d(3,1,1)(cutout_mse_mask.float()).detach()
            cutout_mse_mask[(cutout_mse_mask>0).detach()] = 1
            cutout_mse_mask = Variable(~cutout_mse_mask.byte().data).float()
            cutout_mse_mask = nn.MaxPool2d(3,1,1)(cutout_mse_mask).detach()
        else:
            cutout_mse_mask = Variable(torch.ones(crack_fold_mse_mask.size())).cuda().float()

        mse_weights = border_mse_mask * crack_fold_mse_mask * cutout_mse_mask
        # reweight for focus areas if necessary
        if mask is not None and args.lambda4 > 1:
            mse_weights.data[mask.data == 1] = mse_weights.data[mask.data == 1] * args.lambda4
            mse_weights.data[mask.data > 1] = mse_weights.data[mask.data > 1] * args.lambda5
                
        err = mse(pred, target)
        merr = err * mse_weights

        if vis:
            save_chunk(np.squeeze(merr.data.cpu().numpy()), log_path + name + 'merr' + str(epoch) + '_' + str(t), norm=False)

        smoothness_mask = torch.max(mask, 2 * (pred == 0).float()) if mask is not None else 2 * (pred == 0).float()

        smoothness_weights = Variable(torch.ones(smoothness_mask.size()).cuda())
        smoothness_weights[(smoothness_mask > 0).data] = args.lambda2 # everywhere we have non-standard smoothness, slightly reduce the smoothness penalty
        smoothness_weights[(smoothness_mask > 1).data] = args.lambda3 # on top of cracks and folds only, significantly reduce the smoothness penalty
        if mask is not None:
            hpred = pred.clone().detach()
            hpred[(mask > 1).detach()] = torch.min(pred).data[0]
        else:
            hpred = pred
        return input_src, input_target, pred, hpred, rfield, torch.mean(merr), residuals, smoothness_weights, mse_weights

    X_test = None
    for idxx, tensor_dict in enumerate(test_loader):
        X = tensor_dict['X']
        X_test = Variable(X).cuda().detach()
        X_test.volatile = True
        if idxx > 1:
            break

    for epoch in range(args.epoch, it):
        print('epoch', epoch)
        if not args.inference_only:
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

                X, mask_stack = tensor_dict['X'], tensor_dict['m']
                # Get inputs
                X = Variable(X, requires_grad=False)
                mask_stack = Variable(mask_stack, requires_grad=False)
                X, mask_stack = X.cuda(), mask_stack.cuda()
                stacks, top, left = aug_stacks([X, mask_stack], padding=padding, jitter=not args.no_jitter)
                X, mask_stack = stacks[0], stacks[1]
                
                errs = []
                penalties = []
                consensus_list = []
                smooth_factor = 1 if trunclayer == 0 and fine_tuning else 0.05
                batch = 3
                for sample_idx, i in enumerate(range(num_targets,X.size(1)-num_targets)):
                    ##################################
                    # RUN SAMPLE FORWARD #############
                    ##################################
                    X_ = X[:,i:i+num_targets+1].detach()

                    if num_targets == 1 and min(torch.var(X_[0,0]).data[0], torch.var(X_[0,1]).data[0]) < args.blank_var_threshold:
                        print "Skipping blank sections", torch.var(X_[0,0]).data[0], torch.var(X_[0,1]).data[0]
                        continue

                    if args.crack_masks is not None or args.fold_masks is not None:
                        mask = mask_stack[0,i]
                        target_mask = mask_stack[0,i+1]
                    else:
                        mask = None
                        target_mask = None

                    if args.num_targets > 1:
                        X_, displaced_masks = displace_slice(X_, 0, [mask, target_mask])
                    else:
                        displaced_masks = [mask, target_mask]

                    a, b, pred_, hpred_, field, err_train, residuals, smoothness_mask, mse_mask = run_sample(X_, displaced_masks[0], displaced_masks[1], train=True, vis=(sample_idx == batch-1) and t % 3 == 0)
                        
                    penalty1 = lambda1 * penalty([field], weights=smoothness_mask)
                    cost = err_train + smooth_factor * torch.sum(penalty1)
                    (cost/2).backward(retain_graph=args.unflow>0)
                    errs.append(err_train.data[0])
                    penalties.append(torch.sum(penalty1).data[0])
                    ##################################
                    
                    ##################################
                    # VISUALIZATION ##################
                    ##################################
                    if (sample_idx == batch-1) and t % 3 == 0:
                        a_, b_ = downsample(trunclayer)(a.unsqueeze(0).unsqueeze(0)), (b.unsqueeze(0).unsqueeze(0) if num_targets == 1 else b.unsqueeze(0))
                        npstack = np.squeeze(torch.cat((reverse_dim(b_,1),pred_,a_), 1).data.cpu().numpy())
                        npstack = (npstack - np.min(npstack)) / (np.max(npstack) - np.min(npstack))
                        nphstack = np.squeeze(torch.cat((reverse_dim(b_,1),hpred_,a_), 1).data.cpu().numpy())
                        nphstack = (nphstack - np.min(nphstack)) / (np.max(nphstack) - np.min(nphstack))
                        if args.crack_masks is not None or args.fold_masks is not None:
                            save_chunk(np.squeeze(mask.data.cpu().numpy()), log_path + name + 'mask' + str(epoch) + '_' + str(t), norm=False)
                            save_chunk(np.squeeze(smoothness_mask.data.cpu().numpy()), log_path + name + 'smask' + str(epoch) + '_' + str(t), norm=False)
                            save_chunk(np.squeeze(mse_mask.data.cpu().numpy()), log_path + name + 'mmask' + str(epoch) + '_' + str(t), norm=False)
                        gif(log_path + name + 'stack' + str(epoch) + '_' + str(t), 255 * npstack)
                        gif(log_path + name + 'hstack' + str(epoch) + '_' + str(t), 255 * nphstack)
                            
                        npfield = field.data.cpu().numpy()
                        display_v(npfield, log_path + name + '_field' + str(epoch) + '_' + str(t))
                        npfield[:,:,:,0] = npfield[:,:,:,0] - np.mean(npfield[:,:,:,0])
                        npfield[:,:,:,1] = npfield[:,:,:,1] - np.mean(npfield[:,:,:,1])
                        display_v(npfield, log_path + name + '_cfield' + str(epoch) + '_' + str(t))
                        display_v([r.data.cpu().numpy() for r in residuals[1:]], log_path + name + '_rfield' + str(epoch) + '_' + str(t))
                        
                        np_sp = np.squeeze(penalty1.data.cpu().numpy())
                        np_sp = (np_sp - np.min(np_sp)) / (np.max(np_sp) - np.min(np_sp))
                        save_chunk(np_sp, log_path + name + 'ferr' + str(epoch) + '_' + str(t), norm=False)
                    ##################################
                    
                    ##################################
                    # RUN SAMPLE BACKWARD ###
                    ##################################
                    X_ = reverse_dim(X[:,i-num_targets+1:i+2],1)
                    mask, target_mask = target_mask, mask

                    if args.num_targets > 1:
                        X_, displaced_masks = displace_slice(X_, 0, [mask, target_mask])
                    else:
                        displaced_masks = [mask, target_mask]
                    a, b, pred_, hpred_, field2, err_train, residuals, smoothness_mask, mse_mask = run_sample(X_, displaced_masks[0], displaced_masks[1], train=True)
                    
                    penalty1 = lambda1 * penalty([field2], weights=smoothness_mask)
                    cost = err_train + smooth_factor * torch.sum(penalty1)
                    (cost/2).backward(retain_graph=args.unflow>0)
                    errs.append(err_train.data[0])
                    penalties.append(torch.sum(penalty1).data[0])
                    ##################################
                    
                    ##################################
                    # CONSENSUS PENALTY COMPUTATION ##
                    ##################################
                    consensus = args.unflow * torch.mean(mse(field, -field2))
                    if args.unflow > 0:
                        consensus.backward()
                    consensus_list.append(consensus.data[0])
                    ##################################
                    
                    optimizer.step()
                    model.zero_grad()

                # Save some info
                if len(errs) > 0:
                    mean_err_train = sum(errs) / len(errs)
                    mean_penalty_train = sum(penalties) / len(penalties)
                    mean_consensus = sum(consensus_list) / len(consensus_list)
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

        if trunclayer > 0:
            continue

        preds = []
        ys = []
        stacks, top, left = aug_stacks([X_test], jitter=False, padding=padding)
        X = stacks[0]
        for i in range(num_targets):
            if i == num_targets - 1:
                preds.append(X[:,-1:])
            else:
                preds.append(X[:, -(num_targets-i):-(num_targets-i-1)])

        ys.append(np.squeeze(preds[-1].data.cpu().numpy()))
        errs = []
        for i in reversed(range(1,X.size()[1] - (num_targets - 1))):
            target_ = torch.cat(preds[-num_targets:], 1)
            if len(preds) >= 3 and args.pred:
                target_ = target_ * args.alpha + prednet(torch.cat(preds[-3:], 1)).squeeze(0) * (1 - args.alpha)
            X_ = torch.cat((X[:,i-1:i], target_), 1)
            a, b, pred, hpred, field, err, residuals, smoothness_mask, mse_mask = run_sample(X_, train=False)
            if args.inference_only and args.archive_fields:
                step = len(residuals) - 1 - skiplayers
                print('Archiving vector fields...')
                save_chunk(np.squeeze(pred.data.cpu().numpy()),  log_path + name + '_pred' + str(i) + '_' + str(epoch))
                display_v(field.data.cpu().numpy() * 2, log_path + name + '_field' + str(i) + '_' + str(epoch))

            errs.append(err.data[0])
            a = np.squeeze(a.data.cpu().numpy())
            preds.append(pred)
            ys.append(a)
        
        if args.inference_only:
            print(np.mean(errs))
        skip = 2 ** (size - 1 - trunclayer)
        gif(log_path + name + 'ys' + str(epoch), np.stack(ys) * 255)
        gif(log_path + name + 'pred' + str(epoch), np.squeeze(torch.cat(preds, 1).data.cpu().numpy()) * 255)
        display_v(field.data[:,::skip,::skip].cpu().numpy(), log_path + name + '_field' + str(epoch))

