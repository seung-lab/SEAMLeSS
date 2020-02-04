import torch
import time
import numpy as np
import six

from .loss import unsupervised_loss
from .residuals import res_warp_img, downsample_residuals, upsample_residuals
from .residuals import combine_residuals
from .blockmatch import block_match

def combine_pre_post(res, pre, post):

    result = combine_residuals(post,
                               combine_residuals(res, pre, is_pix_res=True),
                               is_pix_res=True)
    return result

def optimize_pre_post_ups(opti_loss, src, tgt, initial_res, sm, lr, num_iter, opt_mode,
                      src_defects,
                      tgt_defects,
                      src_large_defects,
                      src_small_defects,
                      opt_params={},
                      crop=128):
    wd = 1e-3
    pred_res = initial_res.clone()
    pred_res.requires_grad = False
    pre_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)
    post_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)

    prev_pre_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)
    prev_post_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)

    trainable = [pre_res, post_res]
    if opt_mode == 'adam':
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif opt_mode == 'sgd':
        optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)

    loss_bundle = {
        'src': src,
        'tgt': tgt,
        'tgt_defects': tgt_defects,
        'src_plastic': torch.zeros_like(src_defects, device=tgt_defects.device),
        'tgt_plastic': torch.zeros_like(src_defects, device=tgt_defects.device),
        'src_edges': torch.zeros_like(src_defects, device=tgt_defects.device),
        'tgt_edges': torch.zeros_like(src_defects, device=tgt_defects.device)
    }
    loss_bundle['src_defects'] = src_defects
    loss_bundle['src_small_defects'] = src_small_defects
    loss_bundle['src_large_defects'] = src_large_defects
    prev_loss = []

    s = time.time()
    loss_bundle['pred_res'] = combine_residuals(post_res,
                                   combine_residuals(pred_res, pre_res, is_pix_res=True),
                                   is_pix_res=True)
    while loss_bundle['pred_res'].shape[-2] < src.shape[-2]:
        loss_bundle['pred_res'] = upsample_residuals(loss_bundle['pred_res'])

    loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)
    #import pdb; pdb.set_trace()
    loss_dict = opti_loss(loss_bundle, crop=crop)
    best_loss = loss_dict['result'].cpu().detach().numpy()
    new_best_ago = 0
    lr_halfed_count = 0
    nan_count = 0
    no_impr_count = 0
    new_best_count = 0
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
    for epoch in range(num_iter):

        loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)
        while loss_bundle['pred_res'].shape[-2] < src.shape[-2]:
            loss_bundle['pred_res'] = upsample_residuals(loss_bundle['pred_res'])
        loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)

        loss_dict = opti_loss(loss_bundle, crop=crop)
        loss_var = loss_dict['result']
        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

        curr_loss = loss_var.cpu().detach().numpy()

        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        if np.isnan(curr_loss):
            nan_count += 1
            lr /= 1.5
            lr_halfed_count += 1
            pre_res = prev_pre_res.clone().detach()
            post_res = prev_post_res.clone().detach()
            post_res.requires_grad = True
            pre_res.requires_grad = True
            trainable = [pre_res, post_res]
            if opt_mode == 'adam':
                optimizer = torch.optim.Adam([pre_res, post_res], lr=lr, weight_decay=wd)
            elif opt_mode == 'sgd':
                optimizer = torch.optim.SGD([pre_res, post_res], lr=lr, **opt_params)
            prev_loss = []
            new_best_ago = 0
        else:

            if not np.isnan(curr_loss) and curr_loss < best_loss:
                prev_pre_res = pre_res.clone()
                prev_post_res = post_res.clone()
                best_loss = curr_loss
                #print ("new best")
                new_best_count += 1
                new_best_ago = 0
            else:
                new_best_ago += 1
                if new_best_ago > 12:
                    #print ("No improvement, reducing lr")
                    no_impr_count += 1
                    lr /= 2
                    lr_halfed_count += 1
                    #pre_res = prev_pre_res.clone().detach()
                    #post_res = prev_post_res.clone().detach()
                    #post_res.requires_grad = True
                    #pre_res.requires_grad = True
                    if opt_mode == 'adam':
                        optimizer = torch.optim.Adam([pre_res, post_res], lr=lr)
                    elif opt_mode == 'sgd':
                        optimizer = torch.optim.SGD([pre_res, post_res], lr=lr, **opt_params)
                    new_best_ago -= 5
                prev_loss.append(curr_loss)

                optimizer.zero_grad()
                loss_var.backward()
                optimizer.step()
            if lr_halfed_count >= 15:
                break


    loss_bundle['pred_res'] = combine_pre_post(pred_res, prev_pre_res, prev_post_res)
    while loss_bundle['pred_res'].shape[-2] < src.shape[-2]:
        loss_bundle['pred_res'] = upsample_residuals(loss_bundle['pred_res'])
    loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)
    loss_dict = opti_loss(loss_bundle, crop=crop)

    e = time.time()
    print ("New best: {}, No impr: {}, NaN: {}, Iter: {}".format(new_best_count, no_impr_count, nan_count, epoch))
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
    print (e - s)
    print ('==========')


    return prev_pre_res, prev_post_res

def optimize_pre_post_multiscale_ups(model, pred_res_start, src, tgt, mips, tgt_defects, src_defects,
        src_large_defects, src_small_defects,
        crop=128, bot_mip=4, max_iter=800,
        sm_keys_to_apply={}, mse_keys_to_apply={}, img_mip=4):

    sm_val = 250e0
    sm_val2 = 250e0
    sm = {
        4: sm_val,
        5: sm_val,
        6: sm_val,
        7: sm_val,
        8: sm_val,
        9: sm_val,
        10: sm_val
    }

    lr = {
        4: 15e-2,
        5: 25e-2,
        6: 25e-2,
        7: 15e-2,
        8: 25e-2,
        9: 25e-2,
        10: 25e-2
    }
    num_iter = {
        4: max_iter,
        5: max_iter,
        6: max_iter,
        7: max_iter,
        8: max_iter,
        9: max_iter,
        10: max_iter
    }

    opt_params = {}
    opt_mode = 'adam'
    sm_mask_factor = 0.00
    opti_losses = {}
    for k, v in six.iteritems(sm):
        opti_losses[k] = unsupervised_loss(smoothness_factor=v, use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      sm_keys_to_apply=sm_keys_to_apply,
                                      mse_keys_to_apply=mse_keys_to_apply
                                      )
    pred_res = pred_res_start.clone()

    for m in mips:
        src_downs = src
        tgt_downs = tgt
        pred_res_downs = pred_res
        src_defects_downs = src_defects
        tgt_defects_downs = tgt_defects
        src_small_defects_downs = src_small_defects
        src_large_defects_downs = src_large_defects

        for i in range (bot_mip - img_mip):
            pred_res_downs = downsample_residuals(pred_res_downs)
            src_downs = torch.nn.functional.avg_pool2d(src_downs, 2)
            tgt_downs = torch.nn.functional.avg_pool2d(tgt_downs, 2)
            src_defects_downs = torch.nn.functional.max_pool2d(src_defects_downs, 2)
            tgt_defects_downs = torch.nn.functional.max_pool2d(tgt_defects_downs, 2)
            src_small_defects_downs = torch.nn.functional.max_pool2d(src_small_defects_downs, 2)
            src_large_defects_downs = torch.nn.functional.max_pool2d(src_large_defects_downs, 2)

        for i in range(m - bot_mip):
            pred_res_downs = downsample_residuals(pred_res_downs)


        '''if m == 6:
            src_downs = test_pyramid.state['up'][str(6)]['output'][0:1, 0:4]
            tgt_downs = test_pyramid.state['up'][str(6)]['output'][0:1, 4:8]
        if m == 5:
            src_downs = test_pyramid.state['up'][str(5)]['output'][0:1, :2]
            tgt_downs = test_pyramid.state['up'][str(5)]['output'][0:1, 2:]
        if m == 4:'''
        if model is not None:
            num_reatures = model.state['up'][str(bot_mip)]['output'].shape[1]
            src_downs = model.state['up'][str(bot_mip)]['output'][0:1, :num_reatures//2]
            tgt_downs = model.state['up'][str(bot_mip)]['output'][0:1, num_reatures//2:]

        normer = torch.nn.GroupNorm(num_groups=src_downs.shape[1], num_channels=src_downs.shape[1], affine=False)
        src_downs = normer(src_downs)
        tgt_downs = normer(tgt_downs)

        #print (src_downs.shape, tgt_downs.shape, pred_res_downs.shape, src_defects_downs.shape, tgt_defects_downs.shape)
        pre_res, post_res = optimize_pre_post_ups(opti_losses[bot_mip], src_downs,
                tgt_downs,
                pred_res_downs,
                sm[m], lr[m],
                num_iter=num_iter[m],
                src_defects=src_defects_downs,
                src_small_defects=src_small_defects_downs,
                src_large_defects=src_large_defects_downs,
                tgt_defects=tgt_defects_downs,
                crop=crop, opt_mode=opt_mode,
                opt_params=opt_params
                )

        pre_res_ups = pre_res.detach()
        post_res_ups = post_res.detach()
        for i in range(m - bot_mip):
            pre_res_ups = upsample_residuals(pre_res_ups)
            post_res_ups = upsample_residuals(post_res_ups)

        pred_res = combine_pre_post(pred_res, pre_res_ups, post_res_ups)
    return pred_res


#pred_res_start = torch.zeros_like(pred_res, device=pred_res.device)
def optimize_metric(model, src, tgt, pred_res_start, tgt_defects=None, src_defects=None, src_small_defects=None,
        src_large_defects=None, max_iter=400):
    start = time.time()

    mse_keys_to_apply = {
        'src': [
            {'name': 'src_large_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0)]},
            {'name': 'src',
             'fm': 0,
             'binarization': {'strat': 'ne', 'value': 0.0},
            'mask_value': 0}
            ],
        'tgt': [
            {'name': 'tgt',
             'fm': 0,
             'binarization': {'strat': 'ne', 'value': 0.0},
            'mask_value': 0}
        ]
    }
    sm_keys_to_apply = {
        'src': [
            {'name': 'src_large_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(2, 0.5)],
             "mask_value": 1e-6}
            ],
        'tgt':[
        ]
    }

    mips = [8, 8]

    if src_defects is not None:
        src_defects = src_defects.squeeze(0)
    else:
        src_defects = torch.zeros_like(src)
    if src_small_defects is not None:
        src_small_defects = src_small_defects.squeeze(0)
    else:
        src_small_defects = torch.zeros_like(src)
    if src_large_defects is not None:
        src_large_defects = src_large_defects.squeeze(0)
    else:
        src_large_defects = torch.zeros_like(src)

    if tgt_defects is not None:
        tgt_defects = tgt_defects.squeeze(0)
    else:
        tgt_defects = torch.zeros_like(tgt)

    pred_res_opt = optimize_pre_post_multiscale_ups(model, pred_res_start, src, tgt, mips,
            src_defects=src_defects,
            tgt_defects=tgt_defects,
            src_small_defects=src_small_defects,
            src_large_defects=src_large_defects,
            crop=128, bot_mip=8, img_mip=8, max_iter=max_iter,
            sm_keys_to_apply=sm_keys_to_apply,
            mse_keys_to_apply=mse_keys_to_apply)

    end = time.time()
    print ("OPTIMIZATION FINISHED. Optimizing time: {0:.2f} sec".format(end - start))
    return pred_res_opt
