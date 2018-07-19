import numpy as np
from helpers import save_chunk, gif, reverse_dim, display_v
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
    return grid, F.grid_sample(grid, field).data.cpu().numpy()

def graph(series, fname):
    plt.plot(series)
    plt.title(fname[fname.rindex('/')+1:])
    plt.show()
    plt.savefig(fname)
    plt.clf()

def show_weights(weights, path):
    weights = np.squeeze(weights.data.cpu().numpy())
    x_slice = weights[weights.shape[0]//2]
    y_slice = weights[:,weights.shape[1]//2]

    graph(x_slice, path.format('x_slice'))
    graph(y_slice, path.format('y_slice'))

    weights[weights.shape[0]//2] = np.mean(weights)
    weights[:,weights.shape[0]//2] = np.mean(weights)

    save_chunk(weights, path.format('weights'), norm=False)
    
def visualize_outputs(path, outputs):
    if outputs is None:
        print('Skipping visualization of empty outputs.')
        return

    src = outputs['input_src'].unsqueeze(0).unsqueeze(0) if 'input_src' in outputs else None
    target = outputs['input_target'].unsqueeze(0).unsqueeze(0) if 'input_target' in outputs else None
    pred = outputs['pred'] if 'pred' in outputs else None
    field = outputs['field'] if 'field' in outputs else None
    rfield = outputs['rfield'] if 'rfield' in outputs else None
    residuals = outputs['residuals'] if 'residuals' in outputs else None
    similarity_error_field = outputs['similarity_error_field'] if 'similarity_error_field' in outputs else None
    smoothness_error_field = outputs['smoothness_error_field'] if 'smoothness_error_field' in outputs else None
    similarity_weights = outputs['similarity_weights'] if 'similarity_weights' in outputs else None
    smoothness_weights = outputs['smoothness_weights'] if 'smoothness_weights' in outputs else None
    hpred = outputs['hpred'] if 'hpred' in outputs else None
    src_mask = outputs['src_mask'] if 'src_mask' in outputs else None
    raw_src_mask = outputs['raw_src_mask'] if 'raw_src_mask' in outputs else None
    target_mask = outputs['target_mask'] if 'target_mask' in outputs else None
    consensus = outputs['consensus_field'] if 'consensus_field' in outputs else None
    
    if src is not None and target is not None and pred is not None:
        stack = np.squeeze(torch.cat((reverse_dim(target,1),pred,src), 1).data.cpu().numpy())
        stack = norm(stack, 255)
        gif(path.format('stack'), stack)
        if hpred is not None:
            hstack = np.squeeze(torch.cat((reverse_dim(target,1),hpred,src), 1).data.cpu().numpy())
            hstack = norm(hstack, 255)
            gif(path.format('hstack'), hstack)

    if field is not None:
        grid, distorted_grid = distortion_lines(field)
        save_chunk(grid, path.format('grid'))
        save_chunk(distorted_grid, path.format('dgrid'))
        
        rfield = rfield.data.cpu().numpy()
        display_v(rfield, path.format('field'))
        display_v(rfield, path.format('cfield'), center=True)

    if consensus is not None:
        cfield = consensus.data.cpu().numpy()
        display_v(cfield, path.format('consensus'))
        
    #if residuals is not None and len(residuals) > 1:
    #    residuals = [r.data.cpu().numpy() for r in residuals[1:]]
    #    display_v(residuals, path.format('rfield'))
    #    display_v(residuals, path.format('crfield'), center=True)

    if similarity_error_field is not None:
        save_chunk(norm(similarity_error_field.data.cpu().numpy()), path.format('similarity_error_field'), norm=False)  

    if smoothness_error_field is not None:
        save_chunk(norm(smoothness_error_field.data.cpu().numpy()), path.format('smoothness_error_field'), norm=False)  

    if similarity_weights is not None:
        show_weights(similarity_weights, path.format('similarity_{}'))

    if smoothness_weights is not None:
        show_weights(smoothness_weights, path.format('smoothness_{}'))
                   
    if src_mask is not None:
        save_chunk(np.squeeze(src_mask.data.cpu().numpy()), path.format('src_mask'), norm=False)

    if raw_src_mask is not None:
        save_chunk(np.squeeze(raw_src_mask.data.cpu().numpy()), path.format('raw_src_mask'), norm=False)
        
    if target_mask is not None:
        save_chunk(np.squeeze(target_mask.data.cpu().numpy()), path.format('target_mask'), norm=False)
