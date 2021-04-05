import torch
import time
import numpy as np
import six

from .loss import unsupervised_loss
from .residuals import res_warp_img, downsample_residuals, upsample_residuals
from .residuals import combine_residuals
from .blockmatch import block_match

def combine_pre_post(res, pre, post):
    if pre.shape[-2] < res.shape[-2]:
        pre = upsample_residuals(pre, factor=res.shape[-2]//pre.shape[-2])
    if post.shape[-2] < res.shape[-2]:
        post = upsample_residuals(post, factor=res.shape[-2]//post.shape[-2])

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
                      crop=256,
                      opt_res_coarsness=0):
    wd = 1e-3
    finish = False
    sdb = False
    while not finish:
        #import pdb; pdb.set_trace()
        finish = True
        pred_res = initial_res.tensor()
        pred_res.requires_grad = False

        pre_res = torch.zeros_like(pred_res, device=pred_res.device, requires_grad=True)
        for _ in range(opt_res_coarsness):
            pre_res = downsample_residuals(pre_res).detach()

        pre_res = torch.zeros_like(pre_res, device=pred_res.device, requires_grad=True)

        post_res = torch.zeros_like(pre_res, device=pred_res.device, requires_grad=True)

        prev_pre_res = torch.zeros_like(pre_res, device=pred_res.device, requires_grad=True)
        prev_post_res = torch.zeros_like(pre_res, device=pred_res.device, requires_grad=True)

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
        loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)
        while loss_bundle['pred_res'].shape[-2] < src.shape[-2]:
            loss_bundle['pred_res'] = upsample_residuals(loss_bundle['pred_res'])

        loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)
        loss_dict = opti_loss(loss_bundle, crop=crop)
        best_loss = loss_dict['result'].cpu().detach().numpy()
        new_best_ago = 0
        lr_halfed_count = 0
        nan_count = 0
        no_impr_count = 0
        new_best_count = 0
        similarity_prop = 0
        similarity_prop_dec_ago = 0
        print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        for epoch in range(num_iter):

            loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)
            while loss_bundle['pred_res'].shape[-2] < src.shape[-2]:
                loss_bundle['pred_res'] = upsample_residuals(loss_bundle['pred_res'])
            loss_bundle['pred_tgt'] = res_warp_img(src, loss_bundle['pred_res'], is_pix_res=True)
            if sdb:
                import pdb; pdb.set_trace()
            loss_dict = opti_loss(loss_bundle, crop=crop)
            loss_var = loss_dict['result']
            #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

            curr_loss = loss_var.cpu().detach().numpy()
            curr_sim_prop = loss_dict['similarity_proportion'].cpu().detach().numpy()

            #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
            if np.isnan(curr_loss):
                if sdb:
                    print ("NAN LOSS")
                    import pdb; pdb.set_trace()
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
                if similarity_prop < curr_sim_prop:
                    similarity_prop_dec_ago += 1
                else:
                    similarity_prop_dec_ago = 0
                if similarity_prop_dec_ago > 20:
                    break
                similarity_prop = curr_sim_prop
                min_improve = 1e-14
                if not np.isnan(curr_loss) and curr_loss + min_improve <= best_loss:
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
                #torch.nn.utils.clip_grad_norm([pre_res, post_res], 4e0)
                pre_res.grad[pre_res.grad != pre_res.grad] = 0
                post_res.grad[post_res.grad != post_res.grad] = 0
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
        src_large_defects, src_small_defects, sm_val,
        crop=256, bot_mip=4, max_iter=800,
        img_mip=4,
        sm_keys_to_apply={}, mse_keys_to_apply={}, start_feature=0):

    #sm_val = 220e0
    #sm_val2 = 220e0
    sm = {
        4: sm_val,
        5: sm_val,
        6: sm_val,
        7: sm_val,
        8: sm_val,
        9: sm_val
    }

    lr = {
        4: 18e-2,
        5: 18e-2,
        6: 18e-2,
        7: 1e-2,
        8: 2e-3,
        9: 1e-3
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

            src_defects_downs = torch.nn.functional.max_pool2d(src_defects_downs, 2)
            tgt_defects_downs = torch.nn.functional.max_pool2d(tgt_defects_downs, 2)
            src_small_defects_downs = torch.nn.functional.max_pool2d(src_small_defects_downs, 2)
            src_large_defects_downs = torch.nn.functional.max_pool2d(src_large_defects_downs, 2)

            #pred_res_downs = downsample_residuals(pred_res_downs)
        #for i in range(m - bot_mip):

        '''if m == 6:
            src_downs = test_pyramid.state['up'][str(6)]['output'][0:1, 0:4]
            tgt_downs = test_pyramid.state['up'][str(6)]['output'][0:1, 4:8]
        if m == 5:
            src_downs = test_pyramid.state['up'][str(5)]['output'][0:1, :2]
            tgt_downs = test_pyramid.state['up'][str(5)]['output'][0:1, 2:]
        if m == 4:'''
        num_reatures = model.state['up'][str(bot_mip)]['output'].shape[1]
        src_downs = model.state['up'][str(bot_mip)]['output'][0:1,
                start_feature:num_reatures//2]
        tgt_downs = model.state['up'][str(bot_mip)]['output'][0:1,
                start_feature + num_reatures//2:]

        normer = torch.nn.GroupNorm(num_groups=src_downs.shape[1], num_channels=src_downs.shape[1], affine=False)
        src_downs = normer(src_downs)
        tgt_downs = normer(tgt_downs)

        #print (src_downs.shape, tgt_downs.shape, pred_res_downs.shape, src_defects_downs.shape, tgt_defects_downs.shape)
        pre_res, post_res = optimize_pre_post_ups(opti_losses[m], src_downs,
                tgt_downs,
                pred_res_downs,
                sm[m], lr[m],
                num_iter=num_iter[m],
                src_defects=src_defects_downs,
                src_small_defects=src_small_defects_downs,
                src_large_defects=src_large_defects_downs,
                tgt_defects=tgt_defects_downs,
                crop=crop, opt_mode=opt_mode,
                opt_params=opt_params,
                opt_res_coarsness=(m - bot_mip)
                )

        pre_res_ups = pre_res.detach()
        post_res_ups = post_res.detach()

        pred_res = combine_pre_post(pred_res, pre_res_ups, post_res_ups)
    return pred_res


#pred_res_start = torch.zeros_like(pred_res, device=pred_res.device)
def optimize_metric(model, src, tgt, pred_res_start, tgt_defects=None, src_defects=None, src_small_defects=None,
        src_large_defects=None, max_iter=400):
    start = time.time()
    do = True
    #    import pdb; pdb.set_trace()
    if not do:
        return pred_res_start
    mse_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(0, 0)],
             "mask_value": 1e-9}
            ],
        'tgt':[
            {'name': 'tgt_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(0, 0)],
             "mask_value": 1e-9}
        ]
    }
    sm_keys_to_apply = {
        'src': [
            {'name': 'src_large_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(8, 0.2), (64, 0.4)],
             "mask_value": 1e-6},
            {'name': 'src_small_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(8, 0.4)],
             "mask_value": 1e-9},
            {'name': 'src',
                'fm': 0,
             'binarization': {'strat': 'gt', 'value': -5.0},
            'mask_value': 1e-9}
            ],
        'tgt':[
        ]
    }
    #mse_keys_to_apply['src'] = []
    #mse_keys_to_apply['tgt'] = []

    mips = [9, 7, 5]

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
            crop=64, bot_mip=5, img_mip=4, max_iter=int(max_iter*1.5),
            sm_keys_to_apply=sm_keys_to_apply,
            mse_keys_to_apply=mse_keys_to_apply,
            sm_val=70e0, start_feature=1)

    num_reatures = model.state['up'][str(4)]['output'].shape[1]
    src_bm = model.state['up'][str(4)]['output'][0:1, num_reatures//2 - 1]
    tgt_bm = model.state['up'][str(4)]['output'][0:1, num_reatures - 1]

    src_bm[..., (src == 0).squeeze()] = 0
    tgt_bm[..., (tgt == 0).squeeze()] = 0
    do_bm = False
    if do_bm:
        s = time.time()
        with torch.no_grad():
            warped_tgt = res_warp_img(src_bm, pred_res_opt, is_pix_res=True)
            tile_size = 128
            tile_step = tile_size * 3 // 4
            max_disp = 16
            refinement_res = block_match(warped_tgt, tgt_bm, tile_size=tile_size,
                                   tile_step=tile_step, max_disp=max_disp,
                                   min_overlap_px=1200, filler="inf", peak_ratio_cutoff=1.2)
            pred_res_opt = combine_residuals(pred_res_opt, refinement_res, is_pix_res=True)

        print ("BM time: {}".format(time.time() - s))

    mse_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0)],
             "mask_value": 1e-9}
            ],
        'tgt':[
            {'name': 'tgt_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0)],
             "mask_value": 1e-9}
        ]
    }
    sm_keys_to_apply = {
        'src': [
            {'name': 'src_large_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0), (3, 2), (64, 0.4)],
             "mask_value": 1e-6},
            {'name': 'src_small_defects',
             'binarization': {'strat': 'value', 'value': 0},
             "coarsen_ranges": [(1, 0), (8, 0.4)],
             "mask_value": 1e-9},
            {'name': 'src',
                'fm': 0,
             'binarization': {'strat': 'gt', 'value': -5.0},
            'mask_value': 1e-9}
            ],
        'tgt':[
        ]
    }
    mips = [6]#6, 5, 4]
    pred_res_opt = optimize_pre_post_multiscale_ups(model, pred_res_opt, src, tgt,
            src_defects=src_defects,
            tgt_defects=tgt_defects,
            src_small_defects=src_small_defects,
            src_large_defects=src_large_defects,
            mips=mips, crop=128, bot_mip=4, img_mip=4,
            max_iter=max_iter//2,
            sm_keys_to_apply=sm_keys_to_apply,
            mse_keys_to_apply=mse_keys_to_apply,
            sm_val=120e0, start_feature=0)
    mips = [4]#6, 5, 4]
    pred_res_opt = optimize_pre_post_multiscale_ups(model, pred_res_opt, src, tgt,
            src_defects=src_defects,
            tgt_defects=tgt_defects,
            src_small_defects=src_small_defects,
            src_large_defects=src_large_defects,
            mips=mips, crop=128, bot_mip=4, img_mip=4,
            max_iter=max_iter,
            sm_keys_to_apply=sm_keys_to_apply,
            mse_keys_to_apply=mse_keys_to_apply,
            sm_val=180e0)

    end = time.time()
    print ("OPTIMIZATION FINISHED. Optimizing time: {0:.2f} sec".format(end - start))
    return pred_res_opt
