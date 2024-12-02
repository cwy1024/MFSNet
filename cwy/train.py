# train代码
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

from MyNet.CRUnet.modelcode.MFSNet import *


# -—-—-—-—-—-—-—-—-—-—-—-—-—-—-
#   设置随机种子:10
# -—-—-—-—-—-—-—-—-—-—-—-—-—-—-
def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
SeedSed(seed=10)
# 2018
# IMG_PATH = glob(r"F:\cwy\dataset\Brats2018\processed\224\trainImage\*")
# MASK_PATH = glob(r"F:\cwy\dataset\Brats2018\processed\224\trainMask\*")
# 2019
# IMG_PATH = glob(r"F:\cwy\dataset\brats2019\proceed\2D\trainImage\*")
# MASK_PATH = glob(r"F:\cwy\dataset\brats2019\proceed\2D\trainMask\*")
# 2021
IMG_PATH = glob(r"F:\cwy\dataset\brats2021\224\trainImage\*")
MASK_PATH = glob(r"F:\cwy\dataset\brats2021\224\trainMask\*")


def parse_args():
    SeedSed(seed=10)
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='MFSNet',
                        help='model name')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        help='model architecture')
    parser.add_argument('--deepsupervision', default=False, type=bool)
    parser.add_argument('--dataset', default="Brats2021",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='numpy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='numpy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        help='loss: ')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=12, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    SeedSed(seed=10)

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    SeedSed(seed=10)
    losses = AverageMeter()
    ious = AverageMeter()
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if args.name == "MFSNet":

            # loss1+loss2
            output1, output2 = model(input)
            loss1 = criterion(output1, target)
            loss2 = criterion(output2, target)
            loss = loss1 + loss2
            iou = iou_score(output1, target)

        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    SeedSed(seed=10)
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.name == "MFSNet":

                # loss1+loss2
                output1, output2 = model(input)
                loss1 = criterion(output1, target)
                loss2 = criterion(output2, target)
                loss = loss1 + loss2
                iou = iou_score(output1, target)

            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


SeedSed(seed=10)
loss_list = []
img_paths = IMG_PATH
mask_paths = MASK_PATH

# 2018,2019,2020,2021
# 第一步：划分训练集和临时集（测试集+验证集）
train_img_paths, temp_img_paths, train_mask_paths, temp_mask_paths = \
    train_test_split(img_paths, mask_paths, test_size=0.3, random_state=2)  # 假设30%的数据用于测试+验证

# 第二步：从临时集中划分出验证集和测试集
val_size = 0.5  # 假设验证集占临时集的一半，即整个数据集的20%
val_img_paths, test_img_paths, val_mask_paths, test_mask_paths = \
    train_test_split(temp_img_paths, temp_mask_paths, test_size=val_size,
                     random_state=2)  # 使用不同的random_state以获得不同的随机性

def main():
    SeedSed(seed=10)
    args = parse_args()
    # args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' % (args.dataset, args.arch)
    if not os.path.exists('models/%s/%s' % (args.name, args.dataset)):
        os.makedirs('models/%s/%s' % (args.name, args.dataset))

    print('模型信息 \n----------------------')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('----------------------')

    with open('models/%s/%s/information.txt' % (args.name, args.dataset), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/%s/information.pkl' % (args.name, args.dataset))

    # define loss function (criterion)
    if args.loss == 'BCEDiceLoss':
        criterion = BCEDiceLoss().cuda()

    cudnn.benchmark = True

    print('image_num:%s' % str(len(img_paths)))
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))
    print("test_num:%s" % str(len(test_img_paths)))

    # create model
    print("=> creating model :%s" % args.name)

    model = FMFNet()
    model = model.cuda()
    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = BraDataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = BraDataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    log = pd.DataFrame(index=[], columns=[
        'epoch ', 'lr ', 'loss ', 'iou ', 'val_loss ', 'val_iou '
    ])

    best_iou = 0
    trigger = 0

    for epoch in range(args.epochs):

        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch + 1,
            args.lr,
            round(train_log['loss'], 4),
            round(train_log['iou'], 4),
            round(val_log['loss'], 4),
            round(val_log['iou'], 4),
        ], index=['epoch ', 'lr ', 'loss ', 'iou ', 'val_loss ', 'val_iou '])

        # 将 tmp 转换为 DataFrame
        tmp_df = tmp.to_frame().T
        # 使用 concat 连接 log 和 tmp_df
        log = pd.concat([log, tmp_df], ignore_index=True)
        log.to_csv('models/%s/%s/log.csv' % (args.name, args.dataset), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/%s/model.pth' % (args.name, args.dataset))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
