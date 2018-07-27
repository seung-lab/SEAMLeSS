import torch
import torch.nn.functional as F
from torch.autograd import Variable

def check_mask(mask, binary):
    assert type(mask) == Variable
    if binary:
        assert torch.max(mask).data[0] <= 1
    assert torch.min(mask).data[0] >= 0

def prep_mask(mask):
    mask = mask.detach()
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    return mask

def union(masks):
    return reduce(torch.max, masks)

def intersect(masks):
    return reduce(torch.min, masks)

def invert(mask):
    check_mask(mask, False)
    if type(mask.data) == torch.FloatTensor or type(mask.data) == torch.cuda.FloatTensor:
        return torch.max(mask) - mask
    else:
        return (torch.max(mask).float() - mask.float()).byte()

def low_pass(mask, radius=1):
    return contract(dilate(mask, radius, binary=False), radius, binary=False, ceil=False, return_sum=False)
    
def dilate(mask, radius, binary=True):
    check_mask(mask, binary)
    sz = mask.size()
    mask = prep_mask(mask)
    if type(mask.data) == torch.FloatTensor or type(mask.data) == torch.cuda.FloatTensor:
        return F.max_pool2d(mask, radius*2+1, stride=1, padding=radius).detach().view(sz)
    else:
        return F.max_pool2d(mask.float(), radius*2+1, stride=1, padding=radius).byte().detach().view(sz)
    
def contract(mask, radius, binary=True, ceil=True, return_sum=False):
    check_mask(mask, binary)
    sz = mask.size()
    mask = prep_mask(mask)
    if type(mask.data) == torch.FloatTensor or type(mask.data) == torch.cuda.FloatTensor:
        contracted = -F.max_pool2d(-mask, radius*2+1, stride=1, padding=radius)
        if ceil:
            contracted = torch.ceil(contracted)
    else:
        contracted = -F.max_pool2d(-(mask.float()), radius*2+1, stride=1, padding=radius)
        if ceil:
            contracted = torch.ceil(contracted)
        contracted = contracted.byte()
    if return_sum:
        return contracted.detach().view(sz), torch.sum(contracted).data[0] <= 0
    else:
        return contracted.detach().view(sz)
