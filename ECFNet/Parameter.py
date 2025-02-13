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
from losses import *
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import sklearn.externals
from BraDataSet import *
import torch.optim as optim
# from dataset import Dataset
from datetime import datetime
from skimage.io import imread
from utils import count_params
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from metrics import dice_coef, batch_iou, mean_iou, iou_score


from MyNet.CRUnet.modelcode.ECFNet import *

model = ECFNet()
input1 = torch.randn(1, 4, 224, 224)  # 示例输入，batch_size=1, channels=3, height=224, width=224

# 使用thop的profile函数计算FLOPs
# flops, params = thop.profile(model, inputs=(input1,input2))
flops, params = thop.profile(model, inputs=(input1,))

# 输出结果
print(f"FLOPs: {flops / 1e9} G")  # 将FLOPs转换为GigaFLOPs (GFLOPs)
print(f"Params: {params / 1e6} M")  # 将参数数量转换为MegaBytes (MB)