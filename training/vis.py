import numpy as np
from helpers import save_chunk, gif, reverse_dim
from helpers import dvl, display_v
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def norm(stack, factor=1):
    return factor * ((stack - np.min(stack)) / (np.max(stack) - np.min(stack)))

def distortion_lines(field, spacing=4, thickness=2):
    grid = Variable(torch.ones((1,1,field.size(-2), field.size(-2)))).cuda()
    for idx in range(thickness):
        grid[:,:,:,idx::spacing] = 0
        grid[:,:,idx::spacing,:] = 0
    return grid, F.grid_sample(grid, field, mode='bilinear').data.cpu().numpy()

def graph(series, fname):
    plt.plot(series)
    plt.title(fname[fname.rindex('/')+1:])
    plt.show()
    plt.savefig(fname)
    plt.clf()

def show_weights(weights, path):
    weights = np.squeeze(weights.data.cpu().numpy()[:1,...])
    x_slice = weights[weights.shape[0]//2]
    y_slice = weights[:,weights.shape[1]//2]

    graph(x_slice, path.format('x_slice'))
    graph(y_slice, path.format('y_slice'))

    weights[weights.shape[0]//2] = np.mean(weights)
    weights[:,weights.shape[0]//2] = np.mean(weights)

    save_chunk(weights, path.format('weights'), norm=False)

def visualize_outputs(path, outputs, skip=['residuals', 'hpred'], verbose=False):
    if verbose:
        if outputs is None:
            print('Skipping visualization of empty outputs to path {}.'.format(path))
        else:
            for k in outputs:
                if k in skip:
                    print('Excluding key {} in outputs for visualization.'.format(k))

    if outputs is None:
        return

    v = lambda k: outputs[k] if k in outputs and k not in skip else None

    if v('aug_src') is not None and v('aug_tgt') is not None:
        if v('pred') is not None:
            stack = np.squeeze(torch.cat((reverse_dim(v('aug_tgt'),1),v('pred'),v('aug_src')), 1).data.cpu().numpy()[:1,...])
        else:
            stack = np.squeeze(torch.cat((reverse_dim(v('aug_tgt'),1),v('aug_src')), 1).data.cpu().numpy()[:1,...])
        stack = norm(stack, 255)
        gif(path.format('stack'), stack)

        if v('hpred') is not None:
            hstack = np.squeeze(torch.cat((reverse_dim(v('aug_tgt'),1),v('hpred'),v('aug_src')), 1).data.cpu().numpy()[:1,...])
            hstack = norm(hstack, 255)
            gif(path.format('hstack'), hstack)

    if v('field') is not None:
        grid, distorted_grid = distortion_lines(v('field')[:1,...])
        save_chunk(grid, path.format('grid'))
        save_chunk(distorted_grid, path.format('dgrid'))

        rfield = v('rfield').data.cpu().numpy()[:1,...]
        display_v(rfield, path.format('field'))
        display_v(rfield, path.format('cfield'), center=True)

    if v('consensus_error_field') is not None:
        cfield = v('consensus_error_field').data.cpu().numpy()[:1,...]
        save_chunk(cfield, path.format('consensus_error_field'), norm=False)

    if v('consensus_field') is not None:
        cfield = v('consensus_field').data.cpu().numpy()[:1,...]
        dvl(cfield, path.format('consensus_field'))

    if v('consensus_field_neg') is not None:
        cfield = v('consensus_field_neg').data.cpu().numpy()[:1,...]
        dvl(cfield, path.format('consensus_field_neg'))

    if v('residuals') is not None and len(v('residuals')) > 1:
        residuals = [r.data.cpu().numpy()[:1,...] for r in v('residuals')[1:]]
        display_v(residuals, path.format('rfield'))
        display_v(residuals, path.format('crfield'), center=True)

    if v('similarity_error_field') is not None:
        save_chunk(norm(v('similarity_error_field').data.cpu().numpy()[:1,...]), path.format('similarity_error_field'), norm=False)

    if v('smoothness_error_field') is not None:
        save_chunk(norm(v('smoothness_error_field').data.cpu().numpy()[:1,...]), path.format('smoothness_error_field'), norm=False)

    if v('similarity_weights') is not None:
        show_weights(v('similarity_weights'), path.format('similarity_{}'))

    if v('smoothness_weights') is not None:
        show_weights(v('smoothness_weights'), path.format('smoothness_{}'))

    if v('src_mask') is not None:
        save_chunk(np.squeeze(v('src_mask').data.cpu().numpy()[:1,...]), path.format('src_mask'), norm=False)

    if v('raw_src_mask') is not None:
        save_chunk(np.squeeze(v('raw_src_mask').data.cpu().numpy()[:1,...]), path.format('raw_src_mask'), norm=False)

    if v('tgt_mask') is not None:
        save_chunk(np.squeeze(v('tgt_mask').data.cpu().numpy()[:1,...]), path.format('tgt_mask'), norm=False)
