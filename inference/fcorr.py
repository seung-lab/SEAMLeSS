import torch

r'''
A mis-alignment indicator metric.

Example usages:
    Comparing single pair of 8x8 images:
      f1,p1 = get_fft_power2(torch.tensor(x[0, :8, :8]))  # 8x8 block in slice 0
      f2,p2 = get_fft_power2(torch.tensor(x[1, :8, :8]))  # 8x8 block in slice 1
      return get_hp_fcorr(f1, p1, f2, p2)
    
    Comparing every pair of adjacent 8x8 slices in a Dx8x8 stack:
      f,p = get_fft_power2(x)  # x is Dx8x8 block as a torch tensor
      rho = get_hp_fcorr(f[:-1,:,:,:], p[:-1,:,:], f[1:,:,:,:], p[1:,:,:])  # 1 slice short 
'''

def get_fft_power2(block):
    r'''
    2D FFT on the last two dimensions, and power of FFT components, in one-sided
     representation and the still redundant components set to 0.
    *Assuming the last two dimensions are the same.*
    The returned FFT has one more dimension added at the last dimension for representing
    the 2 channels of complex numbers. The returned power has same number of dimensions as input.
    '''
    f = torch.rfft(block, 2, normalized=True, onesided=True) # currently 2-channel tensor rather than "ComplexFloat"
    # Remove redundant components in one-sided DFT (avoid double counting them)
    onesided = f.shape[-2]   # get the number of non-redundant components from the "last" dim,
              # note this is only valid because our "last" two dims are the same,
    	      # also note the real last dim for array f is for complex number
    f[..., onesided:, 0, :] = 0
    f[..., onesided:, -1, :] = 0
    p = torch.sum(f*f, dim=-1)
    return f,p

def cut_low_freq(fmask, cutoff_1d = 0, cutoff_2d = 0):
    fmask[..., 0:1+cutoff_1d, :] = 0
    if cutoff_1d>0:
        fmask[..., -cutoff_1d:, :] = 0
    fmask[..., :, 0:1+cutoff_1d] = 0

    fmask[..., 0:1+cutoff_2d, 0:1+cutoff_2d] = 0
    if cutoff_2d>0:
        fmask[..., -cutoff_2d:, 0:1+cutoff_2d] = 0

    return fmask

def masked_corr_coef(a, b, mask, n_thres = 2, fill_value = 2):
    r'''
    Correlation coeff applied on last 2+1(spatial + complex) dimensions, 
    only considering elements specified by the mask.
    `mask` should have one less dimension (no complex number channels).
    Return value keeps all dimensions except the last complex channel dim.
    '''
    floatmask = mask.to(a)[...,None]
    N2 = floatmask.sum(dim=(-3,-2,-1), keepdim=True)*2  # *2: two channels of complex number

    an = a - (floatmask * a).sum(dim=(-3, -2,-1), keepdim=True) / N2
    bn = b - (floatmask * b).sum(dim=(-3, -2,-1), keepdim=True) / N2
    an[~mask] = 0   # an = an*floatmask, if it's faster - actually that might be better (boolean
            #mask indexing doesn't broadcast and I had to require `mask` arg to be 1 dimention short)
    bn[~mask] = 0

    dotproduct = (an * bn).sum(dim=(-3,-2,-1), keepdim=True)
    # future:
    # norm() on pytorch master seems able to take vector value for 'dim' argument now
    rho = dotproduct / (an.flatten(start_dim=-3, end_dim=-1).norm(2, dim=-1) *
                        bn.flatten(start_dim=-3, end_dim=-1).norm(2, dim=-1))[...,None,None,None]

    rho[N2<2*n_thres] = fill_value
    rho.squeeze_(-1)  # remove dim for complex channel
    
    return rho

def corr_coef(a, b):
    an = a - a.mean()
    bn = b - b.mean()
    rho = an.dot(bn) / (an.norm(2) * bn.norm(2))
    return rho

def get_hp_fcorr(f1, p1, f2, p2, fill_value = 2, scaling = 256, n_thres = 2):
    r'''
    Correlation coeffecient on high passed and high power frequency components.
    Assuming the frequency domain inputs came from (...x)8x8 image blocks.
    `scaling`:
       Affects the power threshold.
       Meant to be proportional to the range of voxel values, and
       the default value of 256 is assuming voxel values in 0-255.
       Suggested value: 8 * sqrt(image1.std() * image2.std())
    `n_thres`:
       Minimum number of frequency comopnents used to compute the correlation.
    General guideline on `scaling` and `n_thres`:
       Try lower thresholds for smoother / more average(-downsample)ed images.
    Returns `fill_value` when/where not enough components satisfy the criteria.
    '''
    blocksize = 8
    #p_thres = (scaling/2*blocksize*0.15)**2  # unnormalized FFT  p_element ~ N_elements
    p_thres = (scaling/2*0.15)**2  # normalized FFT
    mpower = p1*p2
    
    valid = mpower > p_thres**2
    
    # ignore low frequency components
    valid = cut_low_freq(valid, cutoff_1d = 0, cutoff_2d = 1)
    
    if 0: # unvectorized version (single pair of slices) as a reference
        N = torch.sum(valid)
        if N >= n_thres:
            # this or the cosine similarity built-in in pyTorch?
            rho = corr_coef(f1[valid].flatten(), f2[valid].flatten())
        else:
            rho = 2
    else: # vectorized version: works on stack of many slices
        rho = masked_corr_coef(f1, f2, valid, n_thres = n_thres, fill_value = fill_value)
        
    return rho

def collapse_fcorr(a):
    """Collapse the post-processed fcorr result to [0,1]
    """
    a[a > 1] = 2 - a[a > 1]
    return a
    
def fcorr_conjunction(images, operators):
    """Elementwise multiplication of fcorr images, -1 operator elements indicate negation 
    """ 
    p = collapse_fcorr(images[0])
    if operators[0] == -1:
        p = 1. - p

    for image, operator in zip(images[1:], operators[1:]):
        d = collapse_fcorr(image)
        if operator == -1:
             d = 1. - d
        p *= d     
    return p 
