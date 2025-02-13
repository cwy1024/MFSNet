
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self, log_ratio_init=0.0, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        # 使用log_ratio来参数化权重，以便它们的和始终为1
        self.log_ratio = nn.Parameter(torch.tensor([log_ratio_init]))
        self.smooth = smooth

    def forward(self, input, target):
        # 计算log_ratio的softmax（或sigmoid）变换，但这里我们只需要两个权重，所以使用特殊的变换
        log_bce_weight = self.log_ratio
        log_dice_weight = -self.log_ratio  # 因为权重之和为1，所以只需要一个参数
        # 为了避免数值问题，使用softmax-like的变换，但这里我们手动处理
        bce_weight = torch.sigmoid(log_bce_weight)
        dice_weight = 1 - bce_weight  # 确保权重之和为1

        # 计算BCE损失
        bce = F.binary_cross_entropy_with_logits(input, target.float(), reduction='mean')

        # 计算Dice损失
        input_sigmoid = torch.sigmoid(input)
        num = target.size(0)  # 批次大小
        input_sigmoid = input_sigmoid.view(num, -1)  # 展平处理
        target = target.view(num, -1)  # 展平处理

        intersection = (input_sigmoid * target).sum(1)
        dice_denominator = (input_sigmoid.sum(1) + target.sum(1))
        dice = (2. * intersection + self.smooth) / (dice_denominator + self.smooth)
        dice_loss = 1 - dice.mean()  # 求平均的Dice损失

        # 计算加权损失
        weighted_bce_loss = bce_weight * bce
        weighted_dice_loss = dice_weight * dice_loss
        total_loss = weighted_bce_loss + weighted_dice_loss
        # print(bce_weight)
        # print(dice_weight)

        return total_loss


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super(BCEDiceLoss, self).__init__()
#
#     def forward(self, input, target):
#         bce = F.cross_entropy(input, target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         # input = input.reshape(num, -1)
#
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return bce + dice

