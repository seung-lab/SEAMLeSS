import torch

def mse_loss(input,target):
    out = torch.pow((input[:,16:240,16:240]-target[:,16:240,16:240]), 2)
    loss = out.mean()
    return loss

def smoothness_penalty(fields, label, order=1):
    factor = lambda f: f.size()[2] / 256
    dx =     lambda f: (f[:,:,1:,:] - f[:,:,:-1,:]) * factor(f)
    dy =     lambda f: (f[:,:,:,1:] - f[:,:,:,:-1]) * factor(f)

    def square(f):
        f = torch.sum(f ** 2, 1)
        f = torch.mul(f, 1-label[:,:f.shape[1],:f.shape[2]] )
        return f

    for idx in range(order):
        fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])  # given k-th derivatives, compute (k+1)-th
    square_errors = map(square, fields) # sum along last axis (x/y channel)

    return sum(map(torch.mean, square_errors))

def loss(xs, ys, Rs, rs, label, lambd=0):
    s = smoothness_penalty([Rs[-1]], label, 2)
    pred = torch.squeeze(ys[-1])
    loss = mse_loss(pred, xs[-1][:,1,:,:])+lambd*s

    return loss
