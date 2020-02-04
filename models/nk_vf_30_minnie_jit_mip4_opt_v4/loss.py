import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import six
from .masks import get_mse_and_smoothness_masks, get_mse_and_smoothness_masks2
from .residuals import combine_residuals

from pdb import set_trace as st

def lap(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,:-2,:], p), 2)
    def dxf(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,2:,:,:], p), 1)
    def dyf(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,2:,:], p), 2)
    fields = map(lambda f: [dx(f), dy(f), dxf(f), dyf(f)], fields)
    fields = map(lambda fl: (sum(fl) / 4.0) ** 2, fields)
    field = sum(map(lambda f: torch.sum(f, -1), fields))
    return field


def jacob(fields):
    def dx(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field


def cjacob(fields):
    def center(f):
        fmean_x, fmean_y = torch.mean(f[:,:,:,0]).data[0], torch.mean(f[:,:,:,1]).data[0]
        fmean = torch.cat((fmean_x * torch.ones((1,f.size(1), f.size(2),1)), fmean_y * torch.ones((1,f.size(1), f.size(2),1))), 3)
        fmean = Variable(fmean).cuda()
        return f - fmean

    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        d = torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
        return center(d)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        d = torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
        return center(d)

    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field


def njacob(fields):
    f = fields[0]
    f2 = torch.tensor(f, requires_grad=True)
    f2[:, :, :, 0] = f[:, :, :, 0] / torch.mean(torch.abs(f[:, :, :, 0]))
    f2[:, :, :, 1] = f[:, :, :, 1] / torch.mean(torch.abs(f[:, :, :, 1]))
    return jacob([f2])


def tv(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.abs(torch.cat(fields, -1)), -1)
    return field


def field_dx(f, forward=False):
    if forward:
        delta = f[:,1:-1,:,:] - f[:,2:,:,:]
    else:
        delta = f[:,1:-1,:,:] - f[:,:-2,:,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 0, 0, 1, 1, 0, 0))
    return result

def field_dy(f, forward=False):
    if forward:
        delta = f[:,:,1:-1,:] - f[:,:,2:,:]
    else:
        delta = f[:,:,1:-1,:] - f[:,:,:-2,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 0, 0, 0, 0))
    return result

def field_dxy(f, forward=False):
    if forward:
        delta = f[:,1:-1,1:-1,:] - f[:,2:,2:,:]
    else:
        delta = f[:,1:-1,1:-1,:] - f[:,:-2,:-2,:]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result


def rigidity_score(field_delta, tgt_length, power=2):
    spring_lengths = torch.sqrt(field_delta[..., 0]**2 + field_delta[..., 1]**2)
    spring_deformations = (spring_lengths - tgt_length).abs() ** power
    return spring_deformations

def pix_identity(size, batch=1, device='cuda'):
    result = torch.zeros((batch, size, size, 2), device=device)
    x = torch.arange(size, device=device)
    result[:, :, :, 1] = x
    result = torch.transpose(result, 1, 2)
    result[:, :, :, 0] = x
    result = torch.transpose(result, 1, 2)
    return result

def rigidity(field, power=2):
    identity = pix_identity(size=field.shape[-2])
    field_abs = field + identity

    result = rigidity_score(field_dx(field_abs, forward=False), 1, power=power)
    result += rigidity_score(field_dx(field_abs, forward=True), 1, power=power)
    result += rigidity_score(field_dy(field_abs, forward=False), 1, power=power)
    result += rigidity_score(field_dy(field_abs, forward=True), 1, power=power)
    result += rigidity_score(field_dxy(field_abs, forward=True), 2**(1/2), power=power)
    result += rigidity_score(field_dxy(field_abs, forward=False), 2**(1/2), power=power)
    result /= 6

    #compensate for padding
    result[..., 0:6, :] = 0
    result[..., -6:, :] = 0
    result[..., :,  0:6] = 0
    result[..., :, -6:] = 0

    return result.squeeze()

def smoothness_penalty(ptype='jacob'):
    def penalty(fields, weights=None):
        if ptype ==     'lap': field = lap(fields)
        elif ptype == 'jacob': field = jacob(fields)
        elif ptype == 'cjacob': field = cjacob(fields)
        elif ptype == 'njacob': field = njacob(fields)
        elif ptype ==    'tv': field = tv(fields)
        elif ptype == 'rig': field = rigidity(fields[0])
        elif ptype == 'linrig': field = rigidity(fields[0], power=1)
        elif ptype == 'rig1.5': field = rigidity(fields[0], power=1.5)
        elif ptype == 'rig3': field = rigidity(fields[0], power=3)
        else: raise ValueError("Invalid penalty type: {}".format(ptype))

        if weights is not None:
            field = field * weights
        return field
    return penalty


def supervised_loss(src_var, tgt_var, true_res,
                    pred_res, pred_tgt, threshold=0):
    diff = torch.abs(true_res - pred_res)
    mask = torch.abs(true_res) >= torch.Tensor([threshold]).cuda()
    masked_diff = diff * mask.float()
    return torch.mean(masked_diff**2)


def get_dataset_loss(model, dataset_loader, loss_fn, mip_in, reverse=True, *args, **kwargs):
    losses = {}
    losses['result'] = []
    for sample in dataset_loader:
        model_run_params = {'level_in': mip_in}
        aligned_bundle = align_sample(model, sample, reverse,
                                     model_run_params=model_run_params)
        if aligned_bundle is not None:
            loss_result = loss_fn(aligned_bundle)
            if isinstance(loss_result, dict):
                for k, v in six.iteritems(loss_result):
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(loss_result[k].cpu().detach().numpy())
            else:
                losses['result'].append(loss_result.cpu().detach().numpy())
    for k in losses.keys():
        losses[k] = np.average(losses[k])
    return losses

def get_dataset_loss_general(model, dataset_loader, loss_fn, run_input_kwarg):
    losses = []
    for sample in dataset_loader:
        run_out_dict = model.run_pair(sample, **run_input_kwarg)
        loss_var = loss_fn(**run_out_dict)
        losses.append(loss_var.cpu().detach().numpy())

    return np.average(losses)

def mask_diff_loss(mult=1.0, crosse=True):
    def mse(pred_mask, true_mask):
        diff = true_mask - pred_mask

        false_negative = diff > 0
        diff[false_negative] = diff[false_negative] * mult

        return torch.mean(diff ** 2)

    def cross_entropy(pred_mask, true_mask):
        weight = true_mask.clone()
        weight[true_mask < 0.5] = 1.0
        weight[true_mask > 0.5] = mult
        return torch.nn.functional.binary_cross_entropy_with_logits(pred_mask,
                        true_mask.unsqueeze(0).unsqueeze(0),
                        #pos_weight=torch.FloatTensor([1.0, mult]))
                        weight=weight.unsqueeze(0).unsqueeze(0))

    if crosse:
        return cross_entropy
    else:
        return mse

def inverter_loss(pred_res, inv_res, is_pix_res=True):
    f = combine_residuals(pred_res, inv_res)
    g = combine_residuals(inv_res, pred_res)
    if is_pix_res:
        f = 2 * f / (f.shape[-2])
        g = 2 * g / (g.shape[-2])
    loss = 0.5 * torch.mean(f**2) + 0.5 * torch.mean(g**2)
    return loss

def similarity_score(bundle,
                     weights=None, norm_mse=True, crop=32):
    tgt = bundle['tgt']
    pred_tgt = bundle['pred_tgt']
    if norm_mse:
        normed_bundle = copy.deepcopy(bundle)
        #tgt = torch.nn.InstanceNorm2d(1)(tgt)
        #pred_tgt = torch.nn.InstanceNorm2d(1)(pred_tgt)
        normed_bundle = augment.SergiyNorm(mask_plastic=False)
        tgt = normed_bundle['tgt']
        pred_tgt = res_warp_img(normed_bundle['src'], normed_bundle['pred_res'], is_pix_res=True)

    mse = ((tgt - pred_tgt)**2)
    if crop != 0:
        mse = mse[..., crop:-crop, crop:-crop]

    if weights is not None:
        weights = weights
        if crop != 0:
            weights = weights[..., crop:-crop, crop:-crop]
        total_mse = torch.sum(mse * weights)
        mask_sum  = torch.sum(weights)
        return total_mse / torch.ones_like(mse).sum()
        if mask_sum == 0:
            return total_mse * 0
        else:
            return total_mse / mask_sum
    else:
        return torch.mean(mse)


def vector_magnitude(src_var, tgt_var, true_res, pred_res, pred_tgt):
    return torch.mean(torch.abs(pred_res))


def vector_similarity_score(src_var, tgt_var, true_res,
                            pred_res, pred_tgt, masks, threshold=0):
    diff = torch.abs(true_res - pred_res)
    mask = torch.abs(true_res) >= torch.Tensor([threshold]).cuda()
    masked_diff = diff * mask.float()[..., :, :]
    mean_diff = torch.mean(torch.abs(diff))
    mean_diff_small = torch.mean(torch.abs(masked_diff[..., 20:-20, 20:-20, :]))
    #print ("Vec True: {}, Vec Pred: {}".format(torch.mean(torch.abs(true_res)), torch.mean(torch.abs(pred_res)))))
    #print ("Vec diff: {}, {}".format(mean_diff,mean_diff_small ))
    return mean_diff


def smoothness_score(bundle, smoothness_type,
                     weights=None, crop=8):
    pixelwise = smoothness_penalty(smoothness_type)([bundle['pred_res']], weights)
    if crop != 0:
        pixelwise = pixelwise[..., crop:-crop, crop:-crop]
    result = torch.mean(pixelwise)
    return result


def x_error(src_var, tgt_var, true_res, pred_res, pred_tgt):
    average_x_error = torch.mean(torch.abs(
        pred_res[:, :, :, 0] - true_res[:, :, :, 0]))
    return average_x_error


def unsupervised_loss(smoothness_factor, smoothness_type='rig', use_defect_mask=False,
                      norm_mse=False, white_threshold=-0.1, reverse=True, coarsen_mse=0,
                      coarsen_smooth=0, coarsen_positive_mse=0, tgt_defects_sm=True,
                      tgt_defects_mse=True, positive_mse_mult=1.0, sm_mask_factor=0,
                      sm_decay_factor=0.5, sm_decay_length=0, sm_keys_to_apply={},
                      mse_keys_to_apply={}):
    def compute_loss(bundle, smoothness_mult=1.0, crop=32):
        loss_dict = {}
        if use_defect_mask:
            mse_mask, smoothness_mask = get_mse_and_smoothness_masks2(bundle,
                    sm_keys_to_apply=sm_keys_to_apply,
                    mse_keys_to_apply=mse_keys_to_apply)
        else:
            mse_mask = None
            smoothness_mask = None

        similarity = similarity_score(bundle,
                                      weights=mse_mask,
                                      norm_mse=norm_mse, crop=crop)
        smoothness = smoothness_score(bundle,
                                      weights=smoothness_mask,
                                      smoothness_type=smoothness_type,
                                      crop=crop)
        result =  similarity + smoothness * smoothness_factor

        loss_dict['result'] = result
        loss_dict['similarity'] = similarity
        loss_dict['smoothness'] = smoothness * smoothness_factor * smoothness_mult
        loss_dict['vec_magnitude'] = torch.mean(torch.abs(bundle['pred_res']))
        loss_dict['vec_sim'] = torch.cuda.FloatTensor([0])
        if 'res' in bundle:
            loss_dict['vec_sim'] = torch.mean(torch.abs(bundle['pred_res'] - bundle['res']))
        return loss_dict
    return compute_loss



