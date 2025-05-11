import torch
import numpy as np
from medpy import metric

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def hausdorff_distance95(y_true, y_pred):

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    # print(np.unique(y_pred_np))
    if np.sum(y_pred_np) == 0:
        return 0.0
    hd95 = metric.binary.hd95(y_pred_np, y_true_np)
    return hd95


def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def calculate_metrics(y_true, y_pred, num_classes):
    dices = []
    HD95  = []
    IOU = []
    # print(np.unique(y_true.cpu().detach().numpy()))
    # print(np.unique(y_pred.cpu().detach().numpy()))
    for cls in range(1,num_classes):
        cls_y_true = (y_true == cls).long()
        cls_y_pred = (y_pred == cls).long()
        # cls_y_true = cls_y_true.numpy()
        if len(cls_y_true==1) == 0:
            dices.append(1)
            HD95.append(0)
            IOU.append(1)
        else:
            dices.append(dice_coefficient(cls_y_true, cls_y_pred))
            HD95.append(hausdorff_distance95(cls_y_true,cls_y_pred))
            IOU.append(iou_score(cls_y_true,cls_y_pred))

    return dices,HD95, IOU


def Dice(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """
    Dice metric for segmentation.

    Args:
        pred (torch.Tensor): (N(batch_size), H, W) size tensor 
        target (torch.Tensor): (N(batch_size), H, W) size tensor

    Returns:
        torch.Tensor: average Dice score
    """
    smooth = 1e-3
    intersection = torch.sum(pred * target, dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2)) + smooth)
    
    return dice.mean()

def IoU(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """
    IoU metric for segmentation.

    Args:
        pred (torch.Tensor): (N(batch_size), H, W) size tensor 
        target (torch.Tensor): (N(batch_size), H, W) size tensor

    Returns:
        torch.Tensor: average IoU score
    """
    smooth = 1e-3
    inter = torch.sum(pred * target, dim=(1, 2))
    union = torch.sum(pred + target, dim=(1, 2)) - inter 
    iou = (inter + smooth) / (union + smooth)
    
    return iou.mean()