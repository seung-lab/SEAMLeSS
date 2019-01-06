from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
import os


def arch_desc_to_str(arch_desc):
    fms = arch_desc['fms']
    result = "fms_" + "_".join(["{}".format(fm) for fm in fms]) + '_end'

    if 'tags' in arch_desc and arch_desc['tags'] != []:
        tags = sorted(arch_desc['tags'])
        result += "_tags_" + "_".join(["{}".format(t) for t in tags]) + '_end'
    return result


def str_to_arch_desc(arch_desc_str):
    tokens = arch_desc_str.split("_")
    curr = 0
    assert tokens[curr] == "fms"
    curr += 1

    arch_desc = {}
    fms = []
    while tokens[curr] != 'end':
        fms.append(int(tokens[curr]))
        curr += 1
    curr += 1
    arch_desc['fms'] = fms

    if curr < len(tokens) and tokens[curr] == 'tags':
        tags = []
        while tokens[curr] != 'end':
            tags.append(tokens[curr])
            curr += 1
        arch_desc['tags'] = tags

    return arch_desc


class Masker(nn.Module):
    def initc(self, m, mult=np.sqrt(60)):
        #m.weight.data *= mult
        nn.init.kaiming_normal_(m.weight.data)

        #m.bias.data = 0
        #nn.init.kaiming_normal_(m.bias.data)


    def __init__(self, fms, k):
        super(Masker, self).__init__()
        self.best_val = 1000000

        p = (k - 1) // 2
        self.layers = []

        for i in range(len(fms) - 1):
            self.layers.append(nn.Conv2d(fms[i], fms[i + 1], k, padding=p))

            if i != len(fms) - 2:
                self.layers.append(nn.LeakyReLU(inplace=True))

        #self.layers.append(torch.nn.Sigmoid())
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq(x)

    def get_all_params(self):
        params = []
        params.extend(self.parameters())
        return params


def create_masker(arch_desc=None, init_path=None):
    model_name = arch_desc_to_str(arch_desc)
    model_path = './checkpoints/{}.pth.tar'.format(model_name)
    print (model_path, os.path.isfile(model_path))
    if os.path.isfile(model_path):
        print ("Loading from checkpoint")
        model = torch.load(model_path)['model']
    elif init_path:
        print ("Reinitializing")
        model = torch.load(init_path)['model']
        model.best_val = 1000
        model.arch_desc = arch_desc
        model.name = model_name
    elif arch_desc:
        print ("Creating new model")
        model = Masker(arch_desc)
    else:
        raise RuntimeError("nither init path nor arch_desc given")
    return model.cuda()


def save_checkpoint(state, name):
    path = "./checkpoints/{}.pth.tar".format(name)
    torch.save(state, path)

