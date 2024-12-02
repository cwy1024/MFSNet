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
# from losses import *
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import sklearn.externals
import torch.optim as optim
# from dataset import Dataset
from datetime import datetime
from skimage.io import imread
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms

torch.backends.cudnn.deterministic = True

# from module.SE import SELayer


def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SeedSed(seed=10)

class ChannelAttention(nn.Module):
    SeedSed(seed=10)
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    SeedSed(seed=10)
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class CBAM(nn.Module):
    SeedSed(seed=10)
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)


    def forward(self, x):
        out = x * self.ca(x)
        out1 = x * self.sa(x)
        return out+out1



# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = CBAM(64)
    input = torch.rand(3, 64, 32, 32)
    output = block(input)
    print(input.size(), output.size())