import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.Conv2D = self.__conv(2, name='Conv2D', in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_2 = self.__conv(2, name='Conv2D_2', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_4 = self.__conv(2, name='Conv2D_4', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_6 = self.__conv(2, name='Conv2D_6', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_8 = self.__conv(2, name='Conv2D_8', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_10 = self.__conv(2, name='Conv2D_10', in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_12 = self.__conv(2, name='Conv2D_12', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_14 = self.__conv(2, name='Conv2D_14', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_16 = self.__conv(2, name='Conv2D_16', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_18 = self.__conv(2, name='Conv2D_18', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_20 = self.__conv(2, name='Conv2D_20', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_22 = self.__conv(2, name='Conv2D_22', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_24 = self.__conv(2, name='Conv2D_24', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_26 = self.__conv(2, name='Conv2D_26', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_28 = self.__conv(2, name='Conv2D_28', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_30 = self.__conv(2, name='Conv2D_30', in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_32 = self.__conv(2, name='Conv2D_32', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_34 = self.__conv(2, name='Conv2D_34', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_36 = self.__conv(2, name='Conv2D_36', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_38 = self.__conv(2, name='Conv2D_38', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_40 = self.__conv(2, name='Conv2D_40', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_42 = self.__conv(2, name='Conv2D_42', in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_44 = self.__conv(2, name='Conv2D_44', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_46 = self.__conv(2, name='Conv2D_46', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_48 = self.__conv(2, name='Conv2D_48', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_50 = self.__conv(2, name='Conv2D_50', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_52 = self.__conv(2, name='Conv2D_52', in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.Conv2D_54 = self.__conv(2, name='Conv2D_54', in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        Conv2D_pad      = F.pad(x, (1, 1, 1, 1))
        Conv2D          = self.Conv2D(Conv2D_pad)
        Tanh            = F.tanh(Conv2D)
        Conv2D_2_pad    = F.pad(Tanh, (1, 1, 1, 1))
        Conv2D_2        = self.Conv2D_2(Conv2D_2_pad)
        Tanh_2          = F.tanh(Conv2D_2)
        Conv2D_4_pad    = F.pad(Tanh_2, (1, 1, 1, 1))
        Conv2D_4        = self.Conv2D_4(Conv2D_4_pad)
        Tanh_4          = F.tanh(Conv2D_4)
        Conv2D_6_pad    = F.pad(Tanh_4, (1, 1, 1, 1))
        Conv2D_6        = self.Conv2D_6(Conv2D_6_pad)
        Tanh_6          = F.tanh(Conv2D_6)
        add_8           = Tanh_6 + Tanh
        Conv2D_8_pad    = F.pad(add_8, (1, 1, 1, 1))
        Conv2D_8        = self.Conv2D_8(Conv2D_8_pad)
        Tanh_8          = F.tanh(Conv2D_8)
        MaxPool         = F.max_pool2d(Tanh_8, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        Conv2D_10_pad   = F.pad(MaxPool, (1, 1, 1, 1))
        Conv2D_10       = self.Conv2D_10(Conv2D_10_pad)
        Tanh_10         = F.tanh(Conv2D_10)
        Conv2D_12_pad   = F.pad(Tanh_10, (1, 1, 1, 1))
        Conv2D_12       = self.Conv2D_12(Conv2D_12_pad)
        Tanh_12         = F.tanh(Conv2D_12)
        Conv2D_14_pad   = F.pad(Tanh_12, (1, 1, 1, 1))
        Conv2D_14       = self.Conv2D_14(Conv2D_14_pad)
        Tanh_14         = F.tanh(Conv2D_14)
        Conv2D_16_pad   = F.pad(Tanh_14, (1, 1, 1, 1))
        Conv2D_16       = self.Conv2D_16(Conv2D_16_pad)
        Tanh_16         = F.tanh(Conv2D_16)
        add_20          = Tanh_16 + Tanh_10
        Conv2D_18_pad   = F.pad(add_20, (1, 1, 1, 1))
        Conv2D_18       = self.Conv2D_18(Conv2D_18_pad)
        Tanh_18         = F.tanh(Conv2D_18)
        MaxPool_2       = F.max_pool2d(Tanh_18, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        Conv2D_20_pad   = F.pad(MaxPool_2, (1, 1, 1, 1))
        Conv2D_20       = self.Conv2D_20(Conv2D_20_pad)
        Tanh_20         = F.tanh(Conv2D_20)
        Conv2D_22_pad   = F.pad(Tanh_20, (1, 1, 1, 1))
        Conv2D_22       = self.Conv2D_22(Conv2D_22_pad)
        Tanh_22         = F.tanh(Conv2D_22)
        Conv2D_24_pad   = F.pad(Tanh_22, (1, 1, 1, 1))
        Conv2D_24       = self.Conv2D_24(Conv2D_24_pad)
        Tanh_24         = F.tanh(Conv2D_24)
        Conv2D_26_pad   = F.pad(Tanh_24, (1, 1, 1, 1))
        Conv2D_26       = self.Conv2D_26(Conv2D_26_pad)
        Tanh_26         = F.tanh(Conv2D_26)
        add_32          = Tanh_26 + Tanh_20
        Conv2D_28_pad   = F.pad(add_32, (1, 1, 1, 1))
        Conv2D_28       = self.Conv2D_28(Conv2D_28_pad)
        Tanh_28         = F.tanh(Conv2D_28)
        ResizeNearestNeighbor = F.interpolate(Tanh_28, size=None, scale_factor=2, mode='nearest')
        Conv2D_30_pad   = F.pad(ResizeNearestNeighbor, (0, 1, 0, 1))
        Conv2D_30       = self.Conv2D_30(Conv2D_30_pad)
        Tanh_30         = F.tanh(Conv2D_30)
        add_38          = Tanh_30 + Tanh_18
        Conv2D_32_pad   = F.pad(add_38, (1, 1, 1, 1))
        Conv2D_32       = self.Conv2D_32(Conv2D_32_pad)
        Tanh_32         = F.tanh(Conv2D_32)
        Conv2D_34_pad   = F.pad(Tanh_32, (1, 1, 1, 1))
        Conv2D_34       = self.Conv2D_34(Conv2D_34_pad)
        Tanh_34         = F.tanh(Conv2D_34)
        Conv2D_36_pad   = F.pad(Tanh_34, (1, 1, 1, 1))
        Conv2D_36       = self.Conv2D_36(Conv2D_36_pad)
        Tanh_36         = F.tanh(Conv2D_36)
        Conv2D_38_pad   = F.pad(Tanh_36, (1, 1, 1, 1))
        Conv2D_38       = self.Conv2D_38(Conv2D_38_pad)
        Tanh_38         = F.tanh(Conv2D_38)
        add_48          = Tanh_38 + Tanh_32
        Conv2D_40_pad   = F.pad(add_48, (1, 1, 1, 1))
        Conv2D_40       = self.Conv2D_40(Conv2D_40_pad)
        Tanh_40         = F.tanh(Conv2D_40)
        ResizeNearestNeighbor_2 = F.interpolate(Tanh_40, size=None, scale_factor=2, mode='nearest')
        Conv2D_42_pad   = F.pad(ResizeNearestNeighbor_2, (0, 1, 0, 1))
        Conv2D_42       = self.Conv2D_42(Conv2D_42_pad)
        Tanh_42         = F.tanh(Conv2D_42)
        add_54          = Tanh_42 + Tanh_8
        Conv2D_44_pad   = F.pad(add_54, (1, 1, 1, 1))
        Conv2D_44       = self.Conv2D_44(Conv2D_44_pad)
        Tanh_44         = F.tanh(Conv2D_44)
        Conv2D_46_pad   = F.pad(Tanh_44, (1, 1, 1, 1))
        Conv2D_46       = self.Conv2D_46(Conv2D_46_pad)
        Tanh_46         = F.tanh(Conv2D_46)
        Conv2D_48_pad   = F.pad(Tanh_46, (1, 1, 1, 1))
        Conv2D_48       = self.Conv2D_48(Conv2D_48_pad)
        Tanh_48         = F.tanh(Conv2D_48)
        Conv2D_50_pad   = F.pad(Tanh_48, (1, 1, 1, 1))
        Conv2D_50       = self.Conv2D_50(Conv2D_50_pad)
        Tanh_50         = F.tanh(Conv2D_50)
        add_64          = Tanh_50 + Tanh_44
        Conv2D_52_pad   = F.pad(add_64, (1, 1, 1, 1))
        Conv2D_52       = self.Conv2D_52(Conv2D_52_pad)
        Tanh_52         = F.tanh(Conv2D_52)
        Conv2D_54_pad   = F.pad(Tanh_52, (1, 1, 1, 1))
        Conv2D_54       = self.Conv2D_54(Conv2D_54_pad)
        Tanh_54         = F.tanh(Conv2D_54)
        return Tanh_54


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
if __name__ == '__main__':
    images = np.zeros((1,1,384,384), dtype=np.float32)
    inp = torch.from_numpy(images )
    model = KitModel('converted_pytorch.npy')
    output_pt = model.forward(inp)
    print(output_pt.mean())
