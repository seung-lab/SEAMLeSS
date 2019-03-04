import torch
import torch.nn as nn
from utilities.helpers import (grid_sample, upsample, downsample,
                               load_model_from_dict, compose_fields,
                               upsample_field, time_function)
import numpy as np


class Model(nn.Module):
    """
    Defines an aligner network.
    This is the main class of the architecture code.

    `height` is the number of levels of the network
    `feature_maps` is the number of feature maps per encoding layer
    """

    def __init__(self, height, skip=0, topskips=0, k=7, *args, **kwargs):
        super().__init__()
        self.pyramid = EPyramid(height, skip, topskips, k)

    def forward(self, src, tgt, skip=0, in_field=None, **kwargs):
        src.unsqueeze_(0)
        tgt.unsqueeze_(0)
        return self.pyramid(src, tgt, skip)

    def load(self, path):
        """
        Loads saved weights into the model
        """
        with path.open('rb') as f:
            weights = torch.load(f)
        load_model_from_dict(self, weights)
        return self

    def save(self, path):
        """
        Saves the model weights to a file
        """
        with path.open('wb') as f:
            torch.save(self.state_dict(), f)


class G(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, k=7, f=nn.LeakyReLU(inplace=True), infm=2):
        super(G, self).__init__()
        p = (k-1)//2
        self.conv1 = nn.Conv2d(infm, 32, k, padding=p)
        self.conv2 = nn.Conv2d(32, 64, k, padding=p)
        self.conv3 = nn.Conv2d(64, 32, k, padding=p)
        self.conv4 = nn.Conv2d(32, 16, k, padding=p)
        self.conv5 = nn.Conv2d(16, 2, k, padding=p)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f,
                                 self.conv3, f,
                                 self.conv4, f,
                                 self.conv5)
        self.initc(self.conv1)
        self.initc(self.conv2)
        self.initc(self.conv3)
        self.initc(self.conv4)
        self.initc(self.conv5)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], 1)
        return self.seq(x).permute(0, 2, 3, 1) / 10


class Enc(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, infm, outfm):
        super(Enc, self).__init__()
        if not outfm:
            outfm = infm
        self.f = nn.LeakyReLU(inplace=True)
        self.c1 = nn.Conv2d(infm, outfm, 3, padding=1)
        self.c2 = nn.Conv2d(outfm, outfm, 3, padding=1)
        self.initc(self.c1)
        self.initc(self.c2)
        self.infm = infm
        self.outfm = outfm
        self.seq = [nn.Sequential(
            self.c1,
            self.f,
            self.c2,
            self.f,
        )]

    def forward(self, src, tgt):
        return self.seq[0](src), self.seq[0](tgt)


class PreEnc(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, outfm=12):
        super(PreEnc, self).__init__()
        self.f = nn.LeakyReLU(inplace=True)
        self.c1 = nn.Conv2d(1, outfm // 2, 7, padding=3)
        self.c2 = nn.Conv2d(outfm // 2, outfm // 2, 7, padding=3)
        self.c3 = nn.Conv2d(outfm // 2, outfm, 7, padding=3)
        self.c4 = nn.Conv2d(outfm, outfm // 2, 7, padding=3)
        self.c5 = nn.Conv2d(outfm // 2, 1, 7, padding=3)
        self.initc(self.c1)
        self.initc(self.c2)
        self.initc(self.c3)
        self.initc(self.c4)
        self.initc(self.c5)
        self.pelist = nn.ModuleList(
            [self.c1, self.c2, self.c3, self.c4, self.c5])

    def forward(self, x):
        outputs = []
        for x_ch in range(x.size(1)):
            out = x[:, x_ch:x_ch+1]
            for idx, m in enumerate(self.pelist):
                out = m(out)
                if idx < len(self.pelist) - 1:
                    out = self.f(out)
            outputs.append(out)
        return torch.cat(outputs, 1)


class EPyramid(nn.Module):
    def __init__(self, size, skip, topskips, k):
        super(EPyramid, self).__init__()
        print('Constructing EPyramid with size {} ({} downsamples)...'
              .format(size, size-1))
        fm_0 = 12
        fm_coef = 6
        self.identities = {}
        self.skip = skip
        self.topskips = topskips
        self.size = size
        enc_outfms = [fm_0 + fm_coef * idx for idx in range(size)]
        enc_infms = [1] + enc_outfms[:-1]
        self.mlist = nn.ModuleList([G(k=k, infm=enc_outfms[level]*2)
                                    for level in range(size)])
        self.up = upsample()
        self.down = downsample(type='max')
        self.enclist = nn.ModuleList([Enc(infm=infm, outfm=outfm)
                                      for infm, outfm
                                      in zip(enc_infms, enc_outfms)])
        self.train_size = 1280
        self.pe = PreEnc(fm_0)
        self.nlevels = self.size - 1 - self.topskips
        self.src_encodings = {}
        self.tgt_encodings = {}

    def forward(self, src, tgt, target_level):
        factor = self.train_size / src.shape[-2]

        for i, module in enumerate(self.enclist):
            src, tgt = module(src, tgt)
            self.src_encodings[i] = src
            self.tgt_encodings[i] = tgt
            src, tgt = self.down(src), self.down(tgt)

        field_so_far = 0
        first_iter = True
        for i in range(self.nlevels, target_level - 1, -1):
            if i >= self.skip and i != 0:  # don't run the lowest aligner
                enc_src, enc_tgt = self.src_encodings[i], self.tgt_encodings[i]
                if not first_iter:
                    enc_src = grid_sample(
                        enc_src,
                        field_so_far, padding_mode='zeros')
                rfield = self.mlist[i](enc_src, enc_tgt) * factor
                if first_iter:
                    field_so_far = rfield
                    first_iter = False
                else:
                    field_so_far = compose_fields(rfield, field_so_far)
            if i != target_level:
                field_so_far = upsample_field(field_so_far, i, i-1)
        return field_so_far
