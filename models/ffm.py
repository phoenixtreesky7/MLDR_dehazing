import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

class MixFFM(nn.Module):
    def __init__(self, channle=256, reduction=4,bias=False, norm_layer=nn.InstanceNorm2d):
        super(MixFFM, self).__init__()
        d = max(int(channle/reduction),4)
        self.conv_squeeze_0 = nn.Sequential(nn.Conv2d(channle, channle, 3, padding=1, bias=bias), norm_layer(channle), nn.ReLU(True))

        #self.global_spatial_avg_pool = nn.AdaptiveAvgPool2d(1)
        global_channel_avg_pool = [nn.Conv2d(channle, d, 4, stride=2, padding=1, bias=bias), norm_layer(d), nn.ReLU(True)]
        global_channel_avg_pool += [nn.Conv2d(d, d, 3, stride=1, padding=1, bias=bias), norm_layer(d), nn.ReLU(True)]
        global_channel_avg_pool += [nn.ConvTranspose2d(d, channle, kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(channle), nn.ReLU(True)]
        self.global_channel_avg_pool = nn.Sequential(*global_channel_avg_pool)

        #self.conv_squeeze_1 = nn.Sequential(nn.Conv2d(channle, d, 1, padding=0, bias=bias), nn.ReLU())
        #self.fcs_f0 = nn.Conv2d(d, channle, kernel_size=1, stride=1,bias=bias)
        #self.fcs_f1 = nn.Conv2d(d, channle, kernel_size=1, stride=1,bias=bias)

        #self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        #self.conv_smooth = ConELUBlock(input_channel, input_channel, (5, 3), padding=(2, 1))

    def forward(self, f1, f2):
        #print('f1', f1.shape)

        f1_pooling_map = self.global_channel_avg_pool(f1)
        f2_pooling_map = self.global_channel_avg_pool(f2)
        f1_pooling_map_sigmoid = self.sigmoid(f1_pooling_map)
        f2_pooling_map_sigmoid = self.sigmoid(f2_pooling_map)
        #print('f1_pooling_map',f1_pooling_map.shape, 'f1_pooling_map_sigmoid',f1_pooling_map_sigmoid.shape)
        feature_fuse_1 = f1*f1_pooling_map_sigmoid + f2*f2_pooling_map_sigmoid + f1 + f2
        feature_fuse_1 = self.conv_squeeze_0(f1_pooling_map)
        #print('feature_fuse_1', feature_fuse_1.shape)
        score_att = 0

        return feature_fuse_1, score_att