# ----------------------------------------------------
#    更换为GELU，添加特征融合+边缘特征+FROM
# ----------------------------------------------------
import os
import time
import math
import torch
import joblib
import random
import warnings
import argparse
import numpy as np
import torchvision
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import sklearn.externals
import torch.optim as optim
# from dataset import Dataset
from datetime import datetime
from skimage.io import imread
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms

from MyNet.CRUnet.module.Edge2 import *
from MyNet.CRUnet.module.cbam import *

def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SeedSed(seed=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# ----------------------------------
#   edge detection
# ----------------------------------

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(sobel_models, input_tensor):
    sobel_x, sobel_y = sobel_models
    g_x = sobel_x(input_tensor)
    g_y = sobel_y(input_tensor)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return g

# unet

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthwiseSeparableConv, self).__init__()
        # 多核逐通道卷积
        self.depth_conv1 = nn.Conv2d(in_channels=in_channel,
                                     out_channels=in_channel,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=in_channel)
        self.depth_conv2 = nn.Conv2d(in_channels=in_channel,
                                     out_channels=in_channel,
                                     kernel_size=3,
                                     dilation=2,
                                     stride=1,
                                     padding=2,
                                     groups=in_channel)
        self.depth_conv3 = nn.Conv2d(in_channels=in_channel,
                                     out_channels=in_channel,
                                     kernel_size=3,
                                     dilation=3,
                                     stride=1,
                                     padding=3,
                                     groups=in_channel)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out1 = self.depth_conv1(input)
        out2 = self.depth_conv2(input)
        out3 = self.depth_conv3(input)
        out = out1 + out2 + out3
        out = self.point_conv(out)
        return out


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(GatedConv2d, self).__init__()

        # 卷积层用于生成常规输出
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # 卷积层用于生成门控信号
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        # 计算常规卷积输出
        conv_output = self.conv(x)

        # 计算门控信号，并通过sigmoid函数激活
        gate_signal = torch.sigmoid(self.gate_conv(x))

        # 将常规输出与门控信号逐元素相乘
        gated_output = conv_output * gate_signal

        return gated_output

class Downsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, self_is=True):
        super(Downsample_block, self).__init__()
        self.is_down = self_is

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        # self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        residual = self.conv1x1(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        y = F.gelu(self.bn2(self.conv2(x)))
        y = y + residual
        if self.is_down:
            x = F.max_pool2d(y, 2, stride=2)
            return x, y
        else:
            return y


class Upsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, i=0):
        super(Upsample_block, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels + out_channels, out_channels)
        # self.conv1 = nn.Conv2d(in_channels+out_channels, out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        B, C, H, W = y.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat((x, y), dim=1)
        residual = self.conv1x1(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x + residual
        return x

class CEFIM(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels1, in_channels2, out_channels):
        super(CEFIM, self).__init__()

        self.conv1=nn.Conv2d(in_channels1 +in_channels2 +32, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.casa=DSACA(out_channels)

    def forward(self, x, y,edge):
        b,c,h,w=x.shape
        y=F.interpolate(y, size=(h,w), mode='bilinear', align_corners=False)

        # 16
        num_splits = 16
        split_size_x_16 = x.size(1) // num_splits
        split_size_y_16 = y.size(1) // num_splits

        # 确保通道数可以被整除
        assert x.size(1) % num_splits == 0, "x的通道数必须是16的倍数"
        assert y.size(1) % num_splits == 0, "y的通道数必须是16的倍数"
        splits_x_16 = torch.split(x, split_size_x_16, dim=1)
        splits_y_16 = torch.split(y, split_size_y_16, dim=1)
        # 存储所有拼接后的张量的列表
        concatenated_tensors = []
        for i in range(num_splits):
            # 拼接 edge, x 的第 i 份, 和 y 的第 i 份
            # 注意：这里假设 edge 的批次大小和空间维度与 x 和 y 相同
            concatenated_tensor = torch.cat([edge, splits_x_16[i], splits_y_16[i]], dim=1)
            # 将拼接后的张量添加到列表中
            concatenated_tensors.append(concatenated_tensor)
        s=torch.cat(concatenated_tensors, dim=1)
        x=F.gelu(self.bn1(self.conv1(s)))
        x=self.casa(x)
        return x

class FROM(nn.Module):
    SeedSed(seed=10)

    def __init__(self):
        super(FEM, self).__init__()

        self.conv1x1_x1 = nn.Conv2d(256, 4, kernel_size=1)
        self.conv1x1_x2 = nn.Conv2d(128, 4, kernel_size=1)
        self.conv1x1_x3 = nn.Conv2d(64, 4, kernel_size=1)

        self.conv1=GatedConv2d(4,4,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = GatedConv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = GatedConv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(12)
        self.conv1x1=nn.Conv2d(108, 3, kernel_size=1)

    def forward(self, x1, x2,x3,y):

        x1=F.interpolate(x1, size=(224,224), mode='bilinear', align_corners=False)
        x1=self.conv1x1_x1(x1)
        x2=F.interpolate(x2, size=(224,224), mode='bilinear', align_corners=False)
        x2=self.conv1x1_x2(x2)
        x3=F.interpolate(x3, size=(224,224), mode='bilinear', align_corners=False)
        x3=self.conv1x1_x3(x3)

        x=F.gelu(self.bn1(self.conv1(x1)))
        x=torch.cat([x,x2],dim=1)
        x=F.gelu(self.bn2(self.conv2(x)))
        x = torch.cat([x, x3], dim=1)
        x = F.gelu(self.bn3(self.conv3(x)))
        x=torch.cat([x,y],dim=1)
        x=self.conv1x1(x)
        return x


class ECFNet(nn.Module):
    SeedSed(seed=10)

    def __init__(self):
        in_chan = 4
        out_chan = 3
        super(ECFNet, self).__init__()

        self.down1 = Downsample_block(in_chan, 32)
        self.down2 = Downsample_block(32, 64)
        self.down3 = Downsample_block(64, 128)
        self.bottle1 = Downsample_block(128, 256, self_is=False)
        self.bottle2 = Downsample_block(256, 256, self_is=False)
        self.up3 = Upsample_block(256, 128)
        self.up2 = Upsample_block(128, 64)
        self.up1 = Upsample_block(64, 32)
        self.outconv = nn.Conv2d(32, out_chan, 1)

        self.conv_x5=nn.Conv2d(in_channels=256,out_channels=3,kernel_size=1)
        self.conv_x6=nn.Conv2d(in_channels=128,out_channels=3,kernel_size=1)
        self.conv_x7=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=1)

        self.sobel_x1, self.sobel_y1 = get_sobel(1, 1)
        self.flair_bn=nn.BatchNorm2d(1)
        self.sobel_x2, self.sobel_y2 = get_sobel(1, 1)
        self.tice_bn=nn.BatchNorm2d(1)

        self.cefim1 = CEFIM(32,64,32)
        self.cefim2 = CEFIM(64,128,64)
        self.cefim3 = CEFIM(128,256,128)

        self.fro=FROM()
    def forward(self, x):

        chunks = torch.chunk(x, chunks=4, dim=1)
        flair=chunks[0]
        t1=chunks[1]
        t1ce=chunks[2]
        t2=chunks[3]
        flair_edge = run_sobel((self.sobel_x1, self.sobel_y1), flair)
        t1ce_edge = run_sobel((self.sobel_x2, self.sobel_y2), t1ce)
        flair_edge=self.flair_bn(flair_edge)
        t1ce_edge=self.tice_bn(t1ce_edge)
        edge_sum=torch.cat((flair_edge,t1ce_edge),dim=1)
        edge1=F.max_pool2d(edge_sum, kernel_size=2, stride=2)
        edge2=F.max_pool2d(edge1, kernel_size=2, stride=2)

        x1, y1 = self.down1(x)
        x2, y2 = self.down2(x1)
        x3, y3 = self.down3(x2)
        x4 = self.bottle1(x3)
        x5 = self.bottle2(x4)

        y3=self.cefim3(y3,x4,edge2)
        y2=self.cefim2(y2,y3,edge1)
        y1=self.cefim1(y1,y2,edge_sum)
        x6 = self.up3(x5, y3)
        x7 = self.up2(x6, y2)
        x7=F.interpolate(x7, size=(224,224), mode='bilinear', align_corners=False)
        x8=torch.cat([x7,y1],dim=1)
        out = self.fro(x5,x6,x7,x8)

        return out


if __name__ == '__main__':
    SeedSed(seed=10)
    input = torch.randn((1, 4, 224, 224)).to(device)
    model = ECFNet().to(device)
    out = model(input)
    print(out.shape)
    print(out)
