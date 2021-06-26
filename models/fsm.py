import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

class Feature_Selection_Model(nn.Module):
    def __init__(self, channle=256, reduction=4,bias=False, norm_layer=nn.InstanceNorm2d):
        super(Feature_Selection_Model, self).__init__()

        self.conv_squeeze_0 = nn.Sequential(nn.Conv2d(channle, channle, 3, padding=1, bias=bias), nn.ReLU())

        self.global_spatial_avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.global_channel_avg_pool = nn.Sequential(nn.Conv2d(channle, 1, 3, padding=1, bias=bias), nn.ReLU())

        d = max(int(channle/reduction),4)
        
        self.conv_squeeze_1 = nn.Sequential(nn.Conv2d(channle, d, 1, padding=0, bias=bias), nn.ReLU())
        self.fcs_f0 = nn.Conv2d(d, channle, kernel_size=1, stride=1,bias=bias)
        self.fcs_f1 = nn.Conv2d(d, channle, kernel_size=1, stride=1,bias=bias)

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        #self.conv_smooth = ConELUBlock(input_channel, input_channel, (5, 3), padding=(2, 1))

    def forward(self, f1, f2):

        #input0_trans = torch.transpose(input0, 1, 3)
        #input1_trans = torch.transpose(input1, 1, 3)
        #input2_trans = torch.transpose(input2, 1, 3)
        #input3_trans = torch.transpose(input3, 1, 3)

        #print('f1', f1.shape)
        feature_fuse_1 = f1 + f2
        feature_fuse_1 = self.conv_squeeze_0(feature_fuse_1)
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling_vector = self.global_spatial_avg_pool(feature_fuse_1)
        #pooling_map = self.global_channel_avg_pool(feature_fuse_1)
        #print('pooling',pooling.shape)
        
        squeeze = self.conv_squeeze_1(pooling_vector)
        #print('squeeze',squeeze.shape)

        score_f0 = self.fcs_f0(squeeze)
        score_f1 = self.fcs_f1(squeeze)

        #print('score_f0',score_f0.shape)

        score_cat = torch.cat((score_f0, score_f1),1)
        #print('score_cat',score_cat.shape)
        score_att = self.softmax(score_cat)
        #print('score_att',score_att.shape)
        score_chunk = torch.chunk(score_att, 2, 1)
        #print('score_chunk',score_chunk[0].shape)

        output_f0 = score_chunk[0] * f1 + f1
        output_f1 = score_chunk[1] * f2

        #print('output_f0',output_f0.shape)

        output = output_f0 + output_f1
        #print('output',output.shape)
        #output = self.conv_smooth(output)

        return output, score_att