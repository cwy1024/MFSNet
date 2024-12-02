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
from MyNet.CRUnet.module.cbam import CBAM

def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SeedSed(seed=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        新增modulation 参数： 是DCNv2中引入的调制标量
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 输出通道是2N
        nn.init.constant_(self.p_conv.weight, 0)  # 权重初始化为0
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:  # 如果需要进行调制
            # 输出通道是N
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)  # 在指定网络层执行完backward（）之后调用钩子函数

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):  # x: (b,c,h,w)
        offset = self.p_conv(x)  # (b,2N,h,w) 学习到的偏移量 2N表示在x轴方向的偏移和在y轴方向的偏移
        if self.modulation:  # 如果需要调制
            m = torch.sigmoid(self.m_conv(x))  # (b,N,h,w) 学习到的N个调制标量

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 如果需要调制
        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # (b,c,h,w,N)
            x_offset *= m  # 为偏移添加调制标量

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset





# FLexMFusion

class EFSM(nn.Module):
    SeedSed(seed=10)

    def __init__(self, channel,in_channels,out_channels,POOL=True):
        super(EFSM, self).__init__()
        self.pool=POOL

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.conv1x3=nn.Conv2d(channel,channel,kernel_size=(1,3),padding=(0,1))
        self.conv3x1=nn.Conv2d(channel,channel,kernel_size=(3,1),padding=(1,0))
        self.bn3_3=nn.BatchNorm2d(channel)
        self.conv1x5=nn.Conv2d(channel,channel,kernel_size=(1,5),padding=(0,2))
        self.conv5x1=nn.Conv2d(channel,channel,kernel_size=(5,1),padding=(2,0))
        self.bn5_5=nn.BatchNorm2d(channel)
        self.conv1x7=nn.Conv2d(channel,channel,kernel_size=(1,7),padding=(0,3))
        self.conv7x1=nn.Conv2d(channel,channel,kernel_size=(7,1),padding=(3,0))
        self.bn7_7=nn.BatchNorm2d(channel)
        # self.conv1x1=nn.Conv2d(channel,channel,kernel_size=1)
        # self.bn1_1=nn.BatchNorm2d(channel)
        self.cbam1=CBAM(channel)
        self.cbam2=CBAM(channel)

    def forward(self, x, y):

        if self.pool:
            y=F.max_pool2d(y,2,stride=2)
        y1=F.relu(self.bn3_3(self.conv3x1(self.conv1x3(y))))
        y2=F.relu(self.bn5_5(self.conv5x1(self.conv1x5(y))))
        y3=F.relu(self.bn7_7(self.conv7x1(self.conv1x7(y))))
        # y4=F.relu(self.bn1_1(self.conv1x1(y)))
        s = y1+y2+y3
        s=self.cbam2(s)
        s = torch.cat([x, s], dim=1)
        s = F.relu(self.bn2(self.conv2(s)))
        return s


class Downsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, shrink=2, yes_no=True):
        super(Downsample_block, self).__init__()
        self.yes_no = yes_no
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.convres = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.convres(x)
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        y = x + residual
        if self.yes_no:
            x = F.max_pool2d(y, 2, stride=2)
            return x, y
        else:
            return y


class Upsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, shrink=2):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.convres = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        residual = self.convres(x)
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = x + residual

        return x


class Fusion(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels):
        super(Fusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=5, padding=2, bias=False)

        self.conv3 = nn.Conv2d(in_channels * 6, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

    def forward(self, x, y):
        s = torch.cat([x, y], dim=1)

        s1 = F.sigmoid(self.conv1(s))
        s1 = s * s1
        s2 = F.sigmoid(self.conv2(s))
        s2 = s * s2
        s = torch.cat([s, s1, s2], dim=1)
        s = F.relu(self.bn1(self.conv3(s)))

        return s


class OFEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OFEM, self).__init__()

        self.conv1=nn.Conv2d(32,3,kernel_size=1)
        self.conv2=nn.Conv2d(64,3,kernel_size=1)
        self.conv3=nn.Conv2d(128,3,kernel_size=1)

        self.conv5x5_1=DeformConv2d(3,3)
        self.conv5x5_2=DeformConv2d(3,3)
        self.bn1=nn.BatchNorm2d(3)
        self.bn2=nn.BatchNorm2d(3)
        self.bn3=nn.BatchNorm2d(3)

        self.outfinal = nn.Conv2d(9, out_channels, 1)


    def forward(self, x1,x2,x3):

        x3=self.conv1(x3)
        x2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)
        x2 = self.conv2(x2)
        x1 = F.interpolate(x1, size=(224, 224), mode='bilinear', align_corners=False)
        x1 = self.conv3(x1)

        x2_2=F.relu(self.bn1(self.conv5x5_1(x2)))
        x1_2=F.relu(self.bn2(self.conv5x5_2(x1)))
        x2_2=x2_2+x1
        x1_2=x1_2+x2

        out=torch.cat([x2_2,x1_2,x3],dim=1)
        out=self.outfinal(out)

        return out

# ---------------------------------------------------------
# yuanshimoxing(guonihe)
# ---------------------------------------------------------
class FMFNet(nn.Module):
    SeedSed(seed=10)

    def __init__(self):
        in_chan = 4
        out_chan = 3
        super(FMFNet, self).__init__()
        # main network
        self.down1 = Downsample_block(2, 32)
        self.down2 = Downsample_block(32, 64)
        self.down3 = Downsample_block(64, 128)
        self.bottom_1 = Downsample_block(128, 256, yes_no=False)
        self.up3 = Upsample_block(256, 128)
        self.up2 = Upsample_block(128, 64)
        self.up1 = Upsample_block(64, 32)
        self.outconv1 = nn.Conv2d(32, out_chan, 1)

        # sideroad network
        self.down1_2 = Downsample_block(2, 32)
        self.down2_2 = Downsample_block(32, 64)
        self.down3_2 = Downsample_block(64, 128)
        self.bottom_2 = Downsample_block(128, 256, yes_no=False)
        self.up3_2 = Upsample_block(256, 128)
        self.up2_2 = Upsample_block(128, 64)
        self.up1_2 = Upsample_block(64, 32)
        self.outconv2 = nn.Conv2d(32, out_chan, 1)

        # Fusion
        self.fu1 = Fusion(32)
        self.fu2 = Fusion(64)
        self.fu3 = Fusion(128)
        self.fu4 = Fusion(256)
        self.fu5 = Fusion(128)
        self.fu6 = Fusion(64)

        #EFSM
        self.efsm1=EFSM(64,192,128)
        self.efsm2=EFSM(32,96,64)
        self.efsm3=EFSM(4,36,32,POOL=False)

        # OFEM
        self.ofem=OFEM(9,3)

    def forward(self, x):
        x1, x2 = torch.split(x, [2, 2], dim=1)

        x2_1, y1 = self.down1(x2)
        x2_2, y2 = self.down2(x2_1)
        x2_3, y3 = self.down3(x2_2)
        x2_4 = self.bottom_1(x2_3)
        x2_5 = self.up3(x2_4, y3)
        x2_6 = self.up2(x2_5, y2)
        x2_7 = self.up1(x2_6, y1)
        x2_8 = self.outconv1(x2_7)

        x1_1, y4 = self.down1_2(x1)
        x1_1_1 = self.fu1(x2_1, x1_1)
        x1_2, y5 = self.down2_2(x1_1_1)
        x1_2_2 = self.fu2(x2_2, x1_2)
        x1_3, y6 = self.down3_2(x1_2_2)
        x1_3_3 = self.fu3(x2_3, x1_3)
        x1_4 = self.bottom_2(x1_3_3)
        x1_4_4 = self.fu4(x2_4, x1_4)
        y6=self.efsm1(y6,y5)
        x1_5 = self.up3_2(x1_4_4, y6)
        x1_5_5 = self.fu5(x2_5, x1_5)
        y5=self.efsm2(y5,y4)
        x1_6 = self.up2_2(x1_5_5, y5)
        x1_6_6 = self.fu6(x2_6, x1_6)
        y4=self.efsm3(y4,x)
        x1_7 = self.up1_2(x1_6_6, y4)
        out=self.ofem(x1_5,x1_6,x1_7)

        return out, x2_8



if __name__ == '__main__':
    SeedSed(seed=10)
    input = torch.randn((1, 4, 224, 224)).to(device)
    model = FMFNet().to(device)
    out1, out2 = model(input)
    print(out1.shape, out2.shape)
