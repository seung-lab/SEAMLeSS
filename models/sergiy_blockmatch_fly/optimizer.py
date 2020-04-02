import torch
import time
import numpy as np

from .loss import unsupervised_loss
from .residuals import res_warp_img, downsample_residuals, upsample_residuals, res_warp_res

def optimize_pre_post_ups(opti_loss, src, tgt, src_defects, tgt_defects, initial_res, sm, lr, num_iter, opt_mode,
                      opt_params={}, crop=2):
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
        'src_defects': src_defects,
        'tgt_defects': tgt_defects,
        'src_plastic': torch.zeros_like(src_defects, device=tgt_defects.device),
        'tgt_plastic': torch.zeros_like(src_defects, device=tgt_defects.device),
        'src_edges': torch.zeros_like(src_defects, device=tgt_defects.device),
        'tgt_edges': torch.zeros_like(src_defects, device=tgt_defects.device)
    }

    prev_loss = []

    s = time.time()
    loss_bundle['pred_res'] = res_warp_res(pred_res, pre_res, is_pix_res=True) + post_res
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

        loss_bundle['pred_res'] = res_warp_res(pred_res, pre_res, is_pix_res=True) + post_res
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
            lr /= 2
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


    loss_bundle['pred_res'] = res_warp_res(pred_res, prev_pre_res, is_pix_res=True) + prev_post_res
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

def optimize_pre_post_multiscale_ups(pred_res_start, src, tgt, src_defects, tgt_defects, mips, crop=2, bot_mip=4, img_mip=4, max_iter=800):
    sm_val = 50e0
    sm_val2 = 800e0
    sm = {
        4:  sm_val,
        5:  sm_val,
        6:  sm_val2,
        7:  sm_val2,
        8:  sm_val2,
        9:  sm_val2
    }

    lr = {
        4: 8e-2,
        5: 8e-2,
        6: 25e-2,
        7: 25e-2,
        8: 25e-2,
        9: 25e-2
    }
    num_iter = {
        4: max_iter,
        5: max_iter,
        6: max_iter,
        7: max_iter,
        8: max_iter,
        9: max_iter
    }

    opt_params = {}
    opt_mode = 'adam'

    opti_losses = {
        4: unsupervised_loss(smoothness_factor=sm[4], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        5: unsupervised_loss(smoothness_factor=sm[5], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        6: unsupervised_loss(smoothness_factor=sm[6], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        7: unsupervised_loss(smoothness_factor=sm[7], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=2,coarsen_smooth=2,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        8: unsupervised_loss(smoothness_factor=sm[8], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=3,coarsen_smooth=3,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        9: unsupervised_loss(smoothness_factor=sm[9], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=4,coarsen_smooth=4,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0)
    }
    pred_res = pred_res_start.clone()
    for m in mips:
        src_downs = src
        tgt_downs = tgt
        pred_res_downs = pred_res
        src_defects_downs = src_defects
        tgt_defects_downs = tgt_defects
        for i in range (bot_mip - img_mip):
            pred_res_downs = downsample_residuals(pred_res_downs)

            src_defects_downs = torch.nn.functional.max_pool2d(src_defects_downs, 2)
            tgt_defects_downs = torch.nn.functional.max_pool2d(tgt_defects_downs, 2)

        for i in range(m - bot_mip):
            pred_res_downs = downsample_residuals(pred_res_downs)


        '''if m == 6:
            src_downs = test_pyramid.state['up'][str(6)]['output'][0:1, 0:4]
            tgt_downs = test_pyramid.state['up'][str(6)]['output'][0:1, 4:8]
        if m == 5:
            src_downs = test_pyramid.state['up'][str(5)]['output'][0:1, :2]
            tgt_downs = test_pyramid.state['up'][str(5)]['output'][0:1, 2:]
        if m == 4:'''

        #print (src_downs.shape, tgt_downs.shape, pred_res_downs.shape, src_defects_downs.shape, tgt_defects_downs.shape)
        pre_res, post_res = optimize_pre_post_ups(opti_losses[m], src_downs, tgt_downs, src_defects_downs,
                                              tgt_defects_downs,
                                              pred_res_downs, sm[m], lr[m], num_iter[m],
                                              crop=crop, opt_mode=opt_mode, opt_params=opt_params)

        pre_res_ups = pre_res.detach()
        post_res_ups = post_res.detach()
        for i in range(m - img_mip):
            pre_res_ups = upsample_residuals(pre_res_ups)
            post_res_ups = upsample_residuals(post_res_ups)

        pred_res = res_warp_res(pred_res, pre_res_ups, is_pix_res=True) + post_res_ups
    return pred_res


#pred_res_start = torch.zeros_like(pred_res, device=pred_res.device)
def optimize(src, tgt, pred_res_start, src_defects, tgt_defects, img_mip=6, max_iter=400):
    mips = [9, 8, 7]
    src_defects = src_defects.squeeze(0)
    tgt_defects = tgt_defects.squeeze(0)
    pred_res_opt = optimize_pre_post_multiscale_ups(pred_res_start, src, tgt, src_defects, tgt_defects, mips, crop=0, bot_mip=6, img_mip=6, max_iter=max_iter)

    return pred_res_opt

