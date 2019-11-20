import torch
from torch.autograd import Variable


def lap(fields, device='cuda'):
    def dx(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2), device=device))
        return torch.cat((p, f[:,1:-1,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2), device=device))
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,:-2,:], p), 2)
    def dxf(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2), device=device))
        return torch.cat((p, f[:,1:-1,:,:] - f[:,2:,:,:], p), 1)
    def dyf(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2), device=device))
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,2:,:], p), 2)
    fields = map(lambda f: [dx(f), dy(f), dxf(f), dyf(f)], fields)
    fields = map(lambda fl: (sum(fl) / 4.0) ** 2, fields)
    field = sum(map(lambda f: torch.sum(f, -1), fields))
    return field

def jacob(fields):
    def dx(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2), device='cuda'))
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2), device='cuda'))
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field

def cjacob(fields):
    def center(f):
        fmean_x, fmean_y = torch.mean(f[:,:,:,0]).item(), torch.mean(f[:,:,:,1]).item()
        fmean = torch.cat((fmean_x * torch.ones((1,f.size(1), f.size(2),1), device='cuda'), fmean_y * torch.ones((1,f.size(1), f.size(2),1), device='cuda')), 3)
        fmean = Variable(fmean).cuda()
        return f - fmean

    def dx(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2), device='cuda'))
        d = torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
        return center(d)
    def dy(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2), device='cuda'))
        d = torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
        return center(d)

    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field

def tv(fields):
    def dx(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2), device='cuda'))
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2), device='cuda'))
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.abs(torch.cat(fields, -1)), -1)
    return field


def field_dx(f, forward=False):
    if forward:
        delta = f[:, 1:-1, :, :] - f[:, 2:, :, :]
    else:
        delta = f[:, 1:-1, :, :] - f[:, :-2, :, :]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 0, 0, 1, 1, 0, 0))
    return result


def field_dy(f, forward=False):
    if forward:
        delta = f[:, :, 1:-1, :] - f[:, :, 2:, :]
    else:
        delta = f[:, :, 1:-1, :] - f[:, :, :-2, :]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 0, 0, 0, 0))
    return result


def field_dxy(f, forward=False):
    if forward:
        delta = f[:, 1:-1, 1:-1, :] - f[:, 2:, 2:, :]
    else:
        delta = f[:, 1:-1, 1:-1, :] - f[:, :-2, :-2, :]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result


def field_dxy2(f, forward=False):
    if forward:
        delta = f[:, 1:-1, 1:-1, :] - f[:, 2:, :-2, :]
    else:
        delta = f[:, 1:-1, 1:-1, :] - f[:, :-2, 2:, :]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result


def rigidity_score(field_delta, tgt_length, power=2):
    spring_lengths = torch.sqrt(field_delta[..., 0] ** 2 + field_delta[..., 1] ** 2)
    spring_deformations = (spring_lengths - tgt_length).abs() ** power
    return spring_deformations


def pix_identity(size, batch=1, device="cuda"):
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
    result += rigidity_score(
        field_dxy(field_abs, forward=True), 2 ** (1 / 2), power=power
    )
    result += rigidity_score(
        field_dxy(field_abs, forward=False), 2 ** (1 / 2), power=power
    )
    result += rigidity_score(
        field_dxy2(field_abs, forward=True), 2 ** (1 / 2), power=power
    )
    result += rigidity_score(
        field_dxy2(field_abs, forward=False), 2 ** (1 / 2), power=power
    )
    result /= 8

    # compensate for padding
    result[..., 0:6, :] = 0
    result[..., -6:, :] = 0
    result[..., :, 0:6] = 0
    result[..., :, -6:] = 0

    return result.squeeze()


def smoothness_penalty(ptype):
    def penalty(fields, weights=None):
        if ptype == "lap":
            field = lap(fields)
        elif ptype == "jacob":
            field = jacob(fields)
        elif ptype == "cjacob":
            field = cjacob(fields)
        elif ptype == "tv":
            field = tv(fields)
        elif ptype == "rig":
            field = rigidity(fields[0])
        elif ptype == "linrig":
            field = rigidity(fields[0], power=1)
        elif ptype == "rig1.5":
            field = rigidity(fields[0], power=1.5)
        elif ptype == "rig3":
            field = rigidity(fields[0], power=3)
        else:
            raise ValueError("Invalid penalty type: {}".format(ptype))

        if weights is not None:
            field = field * weights
        return field
    return penalty

def similarity_score(should_reduce=False):
    return lambda x, y: torch.mean((x-y)**2) if should_reduce else (x-y)**2
