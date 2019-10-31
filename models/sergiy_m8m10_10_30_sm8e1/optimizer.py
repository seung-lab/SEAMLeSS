import torch
import time
import numpy as np

from .loss import unsupervised_loss
from .residuals import res_warp_img

def optimizer(src, tgt, pred_res):
    opti_loss = unsupervised_loss(smoothness_factor=3e1, use_defect_mask=False,
                                  white_threshold=-10, reverse=True,
                                  coarsen_mse=0, coarsen_smooth=1,
                                  coarsen_positive_mse=0,
                                  positive_mse_mult=0)
    loss_bundle = {
            'src': src,
            'tgt': tgt,
            'pred_res': pred_res,
            'pred_tgt': res_warp_img(src, pred_res, is_pix_res=True),
            'src_defects': torch.zeros_like(src),
            'tgt_defects': torch.zeros_like(tgt)
    }

    num_epochs = 200
    lr = 1e-2
    loss_bundle['pred_res'].requires_grad = True

    trainable = [loss_bundle['pred_res']]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=0)

    prev_loss = []

    s = time.time()
    for epoch in range(num_epochs):
        loss_bundle['pred_res'] = pred_res
        loss_bundle['pred_tgt'] = res_warp_img(src, pred_res, is_pix_res=True)
        loss_dict = opti_loss(loss_bundle)
        #print (loss_dict)
        loss_var = loss_dict['result']
        curr_loss = loss_var.cpu().detach().numpy()
        optimizer.zero_grad()
        loss_var.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print (curr_loss)

        prev_loss.append(curr_loss)
        if np.isnan(curr_loss):
            lr /= 1.1
            print ("is nan, unding")
            optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=0)
            prev_loss = []
            pred_res = prev_pred_res.clone()
        else:
            prev_pred_res = pred_res.clone()
            if len(prev_loss) >= 30:
                if prev_loss[0] < prev_loss[-1]:
                    lr /= 1.1
                    print ("worse")
                    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=0)
                    prev_loss = []
                else:
                    del prev_loss[0]

    return prev_pred_res
