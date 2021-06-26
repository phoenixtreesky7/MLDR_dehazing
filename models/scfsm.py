import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

class SpatialChannelFFM(nn.Module):
    def __init__(self, channle=256, reduction=4,bias=False, norm_layer=nn.InstanceNorm2d):
        super(SpatialChannelFFM, self).__init__()
        d = max(int(channle/reduction),4)
        self.conv_squeeze_0 = nn.Sequential(nn.Conv2d(channle, channle, 3, padding=1, bias=bias), nn.ReLU(True))

        self.global_spatial_avg_pool = nn.AdaptiveAvgPool2d(1)
        global_channel_avg_pool = [nn.Conv2d(channle, d, 4, stride=2, padding=1, bias=bias), norm_layer(d), nn.ReLU(True)]
        global_channel_avg_pool += [nn.Conv2d(d, d, 3, stride=1, padding=1, bias=bias), norm_layer(d), nn.ReLU(True)]
        global_channel_avg_pool += [nn.ConvTranspose2d(d, channle, kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(channle), nn.ReLU(True)]
        self.global_channel_avg_pool = nn.Sequential(*global_channel_avg_pool)

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
        

        ## spatial attention

        f1_pooling_map = self.global_channel_avg_pool(f1)
        f2_pooling_map = self.global_channel_avg_pool(f2)
        f1_pooling_map_sigmoid = self.sigmoid(f1_pooling_map)
        f2_pooling_map_sigmoid = self.sigmoid(f2_pooling_map)
        #print('f1_pooling_map',f1_pooling_map.shape, 'f1_pooling_map_sigmoid',f1_pooling_map_sigmoid.shape)
        feature_spatial_enhance_1 = f1*f1_pooling_map_sigmoid + f1
        feature_spatial_enhance_2 =  f2*f2_pooling_map_sigmoid + f2

        

        feature_spatial_fuse = feature_spatial_enhance_1 + feature_spatial_enhance_2
        feature_spatial_fuse = self.conv_squeeze_0(feature_spatial_fuse)
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling_vector = self.global_spatial_avg_pool(feature_spatial_fuse)
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

        output_f0 = score_chunk[0] * feature_spatial_enhance_1 + feature_spatial_enhance_1
        output_f1 = score_chunk[1] * feature_spatial_enhance_2 + feature_spatial_enhance_2

        output = output_f0 + output_f1
        #print('output',output.shape)
        #output = self.conv_smooth(output)

        return output, score_att