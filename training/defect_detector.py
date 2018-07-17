import torch
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import morphology
import numpy as np
from helpers import save_chunk
import masks

class DefectDetector(object):
    def __init__(self, net, minor_dilation_radius=1, major_dilation_radius=75, sigmoid_threshold=0.4, cc_count=250):
        for p in net.parameters():
            p.requires_grad = False
        self.net = net
        self.minor_dilation_radius = minor_dilation_radius
        self.major_dilation_radius = major_dilation_radius
        self.sigmoid_threshold = sigmoid_threshold
        self.cc_count = cc_count

    def net_preprocess(self, stack):
        weights = np.array(
            [[1./48, 1./24, 1./48],
             [1./24, 3./4,  1./24],
             [1./48, 1./24, 1./48]]
        )

        kernel = Variable(torch.FloatTensor(weights).expand(10,1,3,3), requires_grad=False).cuda()
        stack = F.conv2d(stack, kernel, padding=1, groups=10) / 255.0

        for idx in range(stack.size(1)):
            stack[:,idx] = stack[:,idx] - torch.mean(stack[:,idx])
            stack[:,idx] = stack[:,idx] / (torch.std(stack[:,idx]) + 1e-6)
        stack = stack.detach()
        stack.volatile = True
        return stack

    def stack_cc(self, stack):
        stack = stack.data.cpu().numpy().astype(bool)
        shape = stack.shape
        stack = np.squeeze(stack)
        stack = np.concatenate([np.expand_dims(morphology.remove_small_objects(stack[i], self.cc_count, connectivity=2), 0) for i in range(stack.shape[0])]).astype(np.uint8)
        stack = stack.reshape(shape)
        return Variable(torch.FloatTensor(stack)).cuda()
    
    def net_postprocess(self, raw_output):
        sigmoided = F.sigmoid(raw_output)
        pooled = F.max_pool2d(sigmoided, self.minor_dilation_radius*2+1, stride=1, padding=self.minor_dilation_radius)
        smoothed = F.avg_pool2d(pooled, self.minor_dilation_radius*2+1, stride=1, padding=self.minor_dilation_radius, count_include_pad=False)
        filtered = self.stack_cc(smoothed > self.sigmoid_threshold)
        dilated = masks.dilate(filtered, self.major_dilation_radius)
        final_output = filtered + dilated
        final_output.volatile = False
        return final_output

    def masks_from_stack(self, stack):
        stack = self.net_preprocess(stack).detach()
        raw_combined_output = torch.cat([torch.max(self.net(stack[:,i:i+1]), 1, keepdim=True)[0] for i in range(stack.size(1))], 1)
        final_output = self.net_postprocess(raw_combined_output)
        return final_output

