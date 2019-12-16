import torch
import time
import numpy as np

from .loss import unsupervised_loss
from .residuals import res_warp_img, downsample_residuals, upsample_residuals, res_warp_res

def optimize_pre_post(opti_loss, src, tgt, src_defects, tgt_defects, initial_res, sm, lr, num_iter):
    pred_res = initial_res.clone()
    pred_res.requires_grad = False
    pre_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)
    post_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)

    prev_pre_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)
    prev_post_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)

    trainable = [pre_res, post_res]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=0)

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
    loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)
    loss_dict = opti_loss(loss_bundle, crop=0)

    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
    for epoch in range(num_iter):
        loss_bundle['pred_res'] = res_warp_res(pred_res, pre_res, is_pix_res=True) + post_res
        loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)

        loss_dict = opti_loss(loss_bundle, crop=0)
        loss_var = loss_dict['result']
        curr_loss = loss_var.cpu().detach().numpy()

        if False and np.isnan(curr_loss):
            lr /= 1.5
            print ("worse")
            pre_res = prev_pre_res.clone().detach()
            post_res = prev_post_res.clone().detach()
            post_res.requires_grad = True
            pre_res.requires_grad = True
            trainable = [pre_res, post_res]
            if opt_mode == 'adam':
                optimizer = torch.optim.Adam([pre_res, post_res], lr=lr)
            elif opt_mode == 'sgd':
                optimizer = torch.optim.SGD([pre_res, post_res], lr=lr, **opt_params)
            prev_loss = []
            continue

        else:
            #print ('step')
            if True:
                #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
                if np.isnan(curr_loss) or (len(prev_loss) >= 30 and curr_loss > prev_loss[0]):
                    lr /= 1.5
                    if np.isnan(curr_loss):
                        #print ('nan')
                        pass

                    #print ("worse")
                    #print (prev_loss)

                    pre_res = prev_pre_res.clone().detach()
                    post_res = prev_post_res.clone().detach()
                    post_res.requires_grad = True
                    pre_res.requires_grad = True
                    trainable = [pre_res, post_res]
                    optimizer = torch.optim.Adam([pre_res, post_res], lr=lr)
                    prev_loss = []
                    continue

                elif len(prev_loss) >= 30:
                    #print ('better')
                    while len(prev_loss) > 0:
                        del prev_loss[0]
                    prev_pre_res = pre_res.clone()
                    prev_post_res = post_res.clone()
            prev_loss.append(curr_loss)

            optimizer.zero_grad()
            loss_var.backward()
            optimizer.step()

    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
    e = time.time()
    print (e - s)

    return pre_res, post_res

def pre_post_multiscale_optimizer(model, src, tgt, pred_res_start, src_defects, tgt_defects):
    sm = {
        4: 60e0,
        5: 80e0,
        6: 80e0,
        7: 80e0,
        8: 80e0,
        9: 80e0
    }

    lr = {
        4: 3e-2,
        5: 8e-2,
        6: 20e-2,
        7: 20e-2,
        8: 20e-2,
        9: 20e-2
    }
    num_iter = {
        4: 500,
        5: 800,
        6: 1000,
        7: 800,
        8: 800,
        9: 800
    }
    src_defects = src_defects.squeeze(0)
    tgt_defects = tgt_defects.squeeze(0)
    opti_losses = {
        4: unsupervised_loss(smoothness_factor=sm[4], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=0,coarsen_smooth=0,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        5: unsupervised_loss(smoothness_factor=sm[5], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=0,coarsen_smooth=0,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        6: unsupervised_loss(smoothness_factor=sm[6], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        7: unsupervised_loss(smoothness_factor=sm[7], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        8: unsupervised_loss(smoothness_factor=sm[8], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0),
        9: unsupervised_loss(smoothness_factor=sm[9], use_defect_mask=True,
                                      white_threshold=-10, reverse=True,
                                      coarsen_mse=1,coarsen_smooth=1,
                                      coarsen_positive_mse=0,
                                      positive_mse_mult=0)


    }
    pred_res = pred_res_start.clone()

    for m in [9, 8, 7, 6, 5, 4]:
        src_downs = src
        tgt_downs = tgt
        pred_res_downs = pred_res
        src_defects_downs = src_defects
        tgt_defects_downs = tgt_defects
        for i in range(m - 4):
            src_downs = torch.nn.functional.avg_pool2d(src_downs, 2)
            tgt_downs = torch.nn.functional.avg_pool2d(tgt_downs, 2)
            pred_res_downs = downsample_residuals(pred_res_downs)
            src_defects_downs = torch.nn.functional.max_pool2d(src_defects_downs, 2)
            tgt_defects_downs = torch.nn.functional.max_pool2d(tgt_defects_downs, 2)


        if m == 6:
            src_downs = model.state['up'][str(6)]['output'][0:1, 0:4]
            tgt_downs = model.state['up'][str(6)]['output'][0:1, 4:8]
        if m == 5:
            src_downs = model.state['up'][str(5)]['output'][0:1, :2]
            tgt_downs = model.state['up'][str(5)]['output'][0:1, 2:]
        if m == 4:
            src_downs = model.state['up'][str(4)]['output'][0:1, :2]
            tgt_downs = model.state['up'][str(4)]['output'][0:1, 2:]

        normer = torch.nn.GroupNorm(num_groups=src_downs.shape[1], num_channels=src_downs.shape[1], affine=False)
        src_downs = normer(src_downs)
        tgt_downs = normer(tgt_downs)
        #import pdb; pdb.set_trace()
        #print (src_downs.shape, tgt_downs.shape, pred_res_downs.shape, src_defects_downs.shape, tgt_defects_downs.shape)
        pre_res, post_res = optimize_pre_post(
                opti_losses[m], src_downs, tgt_downs, src_defects_downs,
                tgt_defects_downs, pred_res_downs, sm[m], lr[m], num_iter[m])

        pre_res_ups = pre_res.detach()
        post_res_ups = post_res.detach()
        for i in range(m - 4):
            pre_res_ups = upsample_residuals(pre_res_ups)
            post_res_ups = upsample_residuals(post_res_ups)

        pred_res = res_warp_res(pred_res, pre_res_ups, is_pix_res=True) + post_res_ups
    return pred_res
