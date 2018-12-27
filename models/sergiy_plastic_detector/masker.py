from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from augment import res_warp_img, res_warp_res
from loss import similarity_score, smoothness_penalty
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


    def __init__(self, arch_desc):
        super(Masker, self).__init__()
        self.best_val = 1000000
        self.name = arch_desc_to_str(arch_desc)

        self.params = {"batchnorm": False,
                       "inputnorm": False,
                       "instancenorm": False,
                       "act_f": nn.LeakyReLU(inplace=True),
                       "k": 7,
                       "initc_mult": np.sqrt(60)}

        fms = arch_desc['fms']

        if 'k' in arch_desc:
            self.params['k'] = arch_desc['k']

        if 'initc_mult' in arch_desc:
            self.params['initc_mult'] = arch_desc['initc_mult']

        if 'act_f' in arch_desc:
            act_f = arch_desc['act_f']
            if act_f == 'lrelu':
                self.params['act_f'] = nn.LeakyReLU(inplace=True)
            elif act_f == 'tanh':
                self.params['act_f'] = nn.Tanh()
            else:
                raise Exception("bad act_f")

        if 'tags' in arch_desc and 'batchnorm' in arch_desc['tags']:
            print ("Use batchnorm!")
            self.params['batchnorm'] = True

        if 'tags' in arch_desc and 'instancenorm' in arch_desc['tags']:
            print ("Use instancenorm!")
            self.params['instancenorm'] = True

        if 'tags' in arch_desc and 'inputnorm' in arch_desc['tags']:
            print ("Use inputnorm!")
            self.params['inputnorm'] = True

        k = self.params['k']
        p = (k - 1) // 2
        self.layers = []

        if self.params['inputnorm']:
            self.layers.append(torch.nn.InstanceNorm2d(num_features=fms[0]))

        for i in range(len(fms) - 1):
            self.layers.append(nn.Conv2d(fms[i], fms[i + 1], k, padding=p))
            self.initc(self.layers[-1], self.params['initc_mult'])

            if i != len(fms) - 2:
                if self.params['batchnorm']:
                    self.layers.append(nn.BatchNorm2d(fms[i + 1]))
                if self.params['instancenorm']:
                    self.layers.append(torch.nn.InstanceNorm2d(
                                                  num_features=fms[i + 1]))
                self.layers.append(self.params['act_f'])

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

