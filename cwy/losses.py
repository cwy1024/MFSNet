
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.cross_entropy(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        # input = input.view(num, -1)       #别的模型用这个
        input = input.reshape(num, -1)  #swin unet用这个

        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
#         """
#         Args:
#             gamma (float): 聚焦参数，通常设置为2.0。
#             alpha (float, optional): 类别权重，默认为None（即均匀权重）。
#             reduction (str): 指定返回的损失类型，'none' | 'mean' | 'sum'。默认为'mean'。
#         """
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         # 将输入通过sigmoid激活函数
#         input = torch.sigmoid(input)
#         # 计算交叉熵损失
#         ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
#         # 计算预测概率
#         pt = torch.exp(-ce_loss)
#         # 计算Focal Loss
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
#
#         if self.alpha is not None:
#             # 如果指定了类别权重，则应用权重
#             alpha_t = self.alpha.gather(0, target.data.view(-1))
#             focal_loss = alpha_t * focal_loss
#
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
#
