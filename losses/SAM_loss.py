"""
Code Reference : SAM-Adapter, https://github.com/tianrun-chen/SAM-Adapter-PyTorch/blob/main/models/iou_loss.py
"""

import torch

class IoULoss(torch.nn.Module):
    def __init__(self):
        """
        IoU Loss for Pytorch 
        Input tensor should be 3-dimensional(N(batch_size), H, W), contains values of either 0(False) or 1(True).
        """
        super(IoULoss, self).__init__()

    def _iou(self, pred, target):
        smooth = 1e-3
        
        inter = (pred * target).sum(dim=(1, 2))
        union = (pred + target).sum(dim=(1, 2)) - inter
        iou = (inter + smooth) / (union + smooth) 
        
        iou = 1 - iou

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)
    
import torch

class DiceLoss(torch.nn.Module):
    def __init__(self):
        """
        Dice Loss for Pytorch
        Input tensor should be 3-dimensional (N(batch_size), H, W), contains values of either 0(False) or 1(True).
        """
        super(DiceLoss, self).__init__()

    def _dice(self, pred, target):
        smooth = 1e-3
        
        inter = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2 * inter + smooth) / (union + smooth)
        
        dice_loss = 1 - dice

        return dice_loss.mean()

    def forward(self, pred, target):
        return self._dice(pred, target)
    
import torch.nn as nn
class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes