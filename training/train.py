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
from torch.utils.data import DataLoader
import h5py

from analysis_helpers import display_v
from pyramid import PyramidTransformer
from dnet import UNet
from helpers import gif, save_chunk, center
from aug import aug_stacks, aug_input, rotate_and_scale, crack

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
    parser.add_argument('--unflow', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--crack', action='store_true')
    parser.add_argument('--num_targets', type=int, default=1)
    parser.add_argument('--lambda1', type=float, default=0.1)
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
    parser.add_argument('--crack_masks', type=str, default=None)
    parser.add_argument('--fold_masks', type=str, default=None)
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

    if args.pred:
        print('Loading prednet...')
        prednet = UNet(3,1).cuda()
        prednet.load_state_dict(torch.load('../framenet/pt/dnet.pt'))
        
    if args.state_archive is None:
        model = PyramidTransformer(size=size, dim=dim, skip=skiplayers, k=kernel_size, dilate=dilate, amp=amp, unet=unet, num_targets=num_targets, name=log_path + name).cuda()
    else:
        model = PyramidTransformer.load(args.state_archive, height=size, dim=dim, skips=skiplayers, k=kernel_size, dilate=dilate, unet=unet, num_targets=num_targets, name=log_path + name)

    for p in model.parameters():
        p.requires_grad = not args.inference_only
    model.train(not args.inference_only)

    train_dataset = StackDataset(os.path.expanduser('~/../eam6/basil_raw_cropped_train_mip5.h5'), os.path.expanduser('~/../eam6/basil_raw_cropped_crack_train_mip5.h5'), basil=True, mask_smooth_factor=31)
    test_dataset = StackDataset(os.path.expanduser('~/../eam6/basil_raw_cropped_train_mip5.h5'), os.path.expanduser('~/../eam6/basil_raw_cropped_crack_train_mip5.h5'), basil=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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

    def run_sample(X, crack_mask=None, train=True):
        model.train(train)
        src, target = X[0,0], torch.squeeze(X[0,1:])
        
        if train and not args.skip_sample_aug:
            # add an artificial crack
            did_crack = False
            if args.crack and crack_mask is None:
                if random.randint(0,1) == 0:
                    did_crack = True
                    src, crack_mask = crack(src, width_range=(8,48))
            
            # random flip
            should_rotate = random.randint(0,1) == 0
            if should_rotate or did_crack:
                src, grid = rotate_and_scale(src.unsqueeze(0).unsqueeze(0), None)
                target = rotate_and_scale(target.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                if crack_mask is not None:
                    crack_mask = rotate_and_scale(crack_mask.unsqueeze(0).unsqueeze(0), grid=grid)[0].squeeze()
                src = src.squeeze()
        
        input_src = src.clone()
        input_target = target.clone()

        if train and not args.skip_sample_aug:
            if random.randint(0,1) == 0:
                input_src = aug_input(input_src)
            if random.randint(0,1) == 0:
                input_target = aug_input(input_target)
        else:
            print 'Skipping sample-wise augmentation...'
                
        pred, field, residuals = model.apply(input_src, input_target, trunclayer)

        # resample some stuff with our new field prediction
        pred = F.grid_sample(downsample(trunclayer)(src.unsqueeze(0).unsqueeze(0)).squeeze().unsqueeze(0).unsqueeze(0), field)
        if crack_mask is not None:
            crack_mask = F.grid_sample(downsample(trunclayer)(crack_mask.unsqueeze(0).unsqueeze(0)), field)

        rfield = field - get_identity_grid(field.size()[-2])
        target = downsample(trunclayer)(target.unsqueeze(0).unsqueeze(0)).squeeze()
        border_mask = -nn.MaxPool2d(5,1,2)(-Variable((pred.data != 0) * (target.data != 0)).float())

        err = mse(pred.squeeze(0).expand(num_targets, pred.size()[-1], pred.size()[-1]), target)
        merr = err * border_mask
        return input_src, target, pred, rfield, torch.mean(merr), residuals, crack_mask, (pred != 0).float()

    X_test = None
    for idxx, (X, mask_test) in enumerate(test_loader):
        X_test = Variable(X).cuda().detach()
        X_test.volatile = True
        if idxx > 1:
            break

    for epoch in range(args.epoch, it):
        print('epoch', epoch)
        if not args.inference_only:
            for t, (X, crack_stack) in enumerate(train_loader):
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
                X = Variable(X, requires_grad=False)
                crack_stack = Variable(crack_stack, requires_grad=False)
                X, crack_stack = X.cuda(), crack_stack.cuda()
                stacks, top, left = aug_stacks([X, crack_stack], padding=padding)
                X, crack_stack = stacks[0], stacks[1]
                
                errs = []
                penalties = []
                consensus_list = []
                smooth_factor = 1 if trunclayer == 0 and fine_tuning else 0.05
                hist_length = 1
                batch = 6
                for sample_idx, i in enumerate(random.sample(range(X.size()[1]-1, num_targets + hist_length - 1, -1), batch//hist_length)):
                    for offset in range(hist_length):
                        X_ = torch.cat((X[:,i:i+1], X[:,i-num_targets-offset:i-offset]), 1)
                        if args.crack_masks is not None:
                            crack_mask = crack_stack[0,i].cuda()
                        else:
                            crack_mask = None
                        a, b, pred_, field, err_train, residuals, crack_mask, border_mask = run_sample(X_.detach(), crack_mask, train=True)
                        
                        penalty1 = lambda1 * penalty([field], crack_mask, border_mask)
                        cost = err_train + smooth_factor * penalty1
                        (cost/2).backward(retain_graph=True)
                        errs.append(err_train.data[0])
                        penalties.append(penalty1.data[0])
                        
                        if sample_idx == batch//hist_length-1 and (t % 4 == 0 or args.state_archive is not None):
                            a_, b_ = downsample(trunclayer)(a.unsqueeze(0).unsqueeze(0)), b.unsqueeze(0).unsqueeze(0)
                            npstack = np.squeeze(torch.cat((a_,b_,pred_), 1).data.cpu().numpy())
                            npstack = (npstack - np.min(npstack)) / (np.max(npstack) - np.min(npstack))
                            if args.crack_masks:
                                save_chunk(np.squeeze(crack_mask.data.cpu().numpy()), log_path + name + 'mask' + str(epoch) + '_' + str(t) + '_' + str(offset))
                            gif(log_path + name + 'stack' + str(epoch) + '_' + str(t) + '_' + str(offset), 255 * npstack)
                            display_v(field.data.cpu().numpy()[:,::8,::8,:], log_path + name + '_field' + str(epoch) + '_' + str(t) + '_' + str(offset))
                            #[display_v(residuals[idx].data.cpu().numpy()[:,::2**(idx),::2**(idx),:], log_path + name + 'rfield' + str(epoch) + '_' + str(t) + '_' + str(idx)) for idx in range(1, len(residuals))]

                        X_ = torch.cat((X[:,i-num_targets-offset:i-num_targets-offset+1], X[:,i-num_targets+1:i+1]), 1)
                        if args.crack_masks is not None:
                            crack_mask = crack_stack[0,i-num_targets-offset].cuda()
                        else:
                            crack_mask = None
                        a, b, pred_, field2, err_train, residuals, crack_mask, border_mask = run_sample(X_.detach(), crack_mask, train=True)
                        
                        penalty1 = lambda1 * penalty([field2], crack_mask, border_mask)
                        cost = err_train + smooth_factor * penalty1
                        (cost/2).backward(retain_graph=True)
                        errs.append(err_train.data[0])
                        penalties.append(penalty1.data[0])

                        consensus = args.unflow * torch.mean(mse(field, -field2))
                        consensus.backward()
                        consensus_list.append(consensus.data[0])
                        
                        optimizer.step()
                        model.zero_grad()

                # Save some info
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
            a, b, pred, field, err, residuals, crack_mask, border_mask = run_sample(X_, train=False)
            penalty([field], crack_mask, border_mask)
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

