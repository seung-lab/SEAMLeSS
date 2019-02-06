import torch

r'''
Example usage:
    f1,p1 = get_fft_power2(torch.tensor(x[:8,:8,0]))  # 8x8 block in slice 1
    f2,p2 = get_fft_power2(torch.tensor(x[:8,:8,1]))  # 8x8 block in slice 1
    return get_hp_fcorr(f1, p1, f2, p2)
'''

def get_fft_power2(block):
    f = torch.rfft(block, 2, normalized=True, onesided=True) # currently 2-channel tensor rather than "ComplexFloat"
    p = torch.sum(f*f, dim=-1) #torch.pow()
    return f,p

def cut_low_freq(fmask, cutoff_1d = 0, cutoff_2d = 0):
    fmask[..., 0:1+cutoff_1d, :] = 0
    fmask[..., :, 0:1+cutoff_1d] = 0
    fmask[..., 0:1+cutoff_2d, 0:1+cutoff_2d] = 0
    return fmask

def corr_coef(a, b):
    an = a - a.mean()
    bn = b - b.mean()
    rho = an.dot(bn) / (an.norm(2) * bn.norm(2))
    return rho

def get_hp_fcorr(f1, p1, f2, p2):
    r'''
    Correlation coeffecient on high passed and high power frequency components
    Assuming 8x8 blocks, voxel value in 0-255
    Returns 2 when not enough components satisfy the criteria.
    '''
    blocksize = 8
    #thres = 256/2*blocksize*0.15  # unnormalized FFT  p_element ~ sqrt(N_elements)
    p_thres = 256/2*0.15  # normalized FFT
    n_thres = 3
    mpowersqrd = p1*p2
    
    valid = mpowersqrd > p_thres**4
    
    # ignore low frequency components
    valid = cut_low_freq(valid, cutoff_1d = 0, cutoff_2d = 1)
    
    N = torch.sum(valid)
    if N > n_thres:
        # this or the cosine similarity built-in in pyTorch?
        rho = corr_coef(f1[valid].flatten(), f2[valid].flatten())
    else:
        rho = 2
        
    return rho
    

