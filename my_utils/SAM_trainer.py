import torch

import numpy as np 
from tqdm import tqdm 
from typing import Tuple
from torchvision.transforms.functional import to_pil_image

from segment_anything.utils.transforms import ResizeLongestSide
from my_utils.Sampling_Combine import max_distance_sample, contour_sample,combine,process_input_SAM, combine_choas, process_input_SAM_Choas,contour_sample_without_bs,combine_cell,process_input_SAM_cell
# from .make_prompt import *
from .metrics import *
from my_utils.metrics import calculate_metrics
from matplotlib import pyplot as plt




def model_train(
    model,
    data_loader,
    criterion,
    optimizer,        
    device,
    scheduler,
    num
) -> Tuple[float, float, float, float]:
    """
    Train the model

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions 
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """
    
    # Training
    model.train()
    
    running_iouloss = 0.0
    running_diceloss = 0.0
    
    running_dice = 0.0
    running_iou = 0.0
    
    n_data = 0
    
    diceloss = criterion[0]    
    iouloss = criterion[1]
    celoss = criterion[2]

    transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
    loss_all = 0.0
    for Sample_List in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels, PLs, scribbles, id = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["pesudo_label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
        labels_np = np.array(labels.cpu())
        
        # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
        
        batched_input_LV, batched_input_MYO, batched_input_RV = [], [], []
        for image, label,PL, scribble in zip(images,labels,PLs, scribbles):
            # prepare image
            original_size = image.shape[1:3]
            image_RGB = torch.cat([image, image, image], dim=0)

            image_RGB = transform.apply_image(image_RGB)
            image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
            image_RGB = image_RGB.permute(2, 0, 1).contiguous()

            sampled_point_batch_LV =  contour_sample_without_bs(scribble, 1, num)
            sampled_point_batch_MYO = contour_sample_without_bs(scribble, 2, num)
            sampled_point_batch_RV =  contour_sample_without_bs(scribble, 3, num)
            sampled_point_batch_background = contour_sample_without_bs(scribble, 4, num)

            all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO = combine(sampled_point_batch_LV, sampled_point_batch_RV, sampled_point_batch_MYO, sampled_point_batch_background)
            batched_input_LV, batched_input_MYO, batched_input_RV = process_input_SAM(transform,image_RGB, original_size, all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO, batched_input_LV, batched_input_MYO, batched_input_RV)
            
            
        batched_inputs = [batched_input_LV, batched_input_MYO, batched_input_RV]
        batched_output_masks,batched_output_masks_pred = -1e5*torch.zeros((images.shape[0],4,images.shape[2],images.shape[3])).cuda(),-1e5*torch.zeros((images.shape[0],4,images.shape[2],images.shape[3])).cuda()
        masks = torch.zeros((images.shape[2],images.shape[3])).cuda()
        masks_pred = torch.zeros((images.shape[2],images.shape[3])).cuda()
        for i, batched_input_cls in enumerate(batched_inputs):
            batched_output = model(batched_input_cls, multimask_output=False)
            for j in range(data_loader.batch_size):
                batched_output_masks[j,i+1,:,:] = batched_output[j]["masks"][0][0]
                # batched_output_masks_pred[j,i+1,:,:] = batched_output[j]["masks_pred"][0][0]
                masks += batched_output_masks[j,i+1,:,:]
            masks = masks/3
            batched_output_masks[0,i+1,:,:] = masks


        iou_loss_ = iouloss(batched_output_masks, PLs.unsqueeze(1))
        dice_loss_ = diceloss(batched_output_masks, PLs.unsqueeze(1)) 
        pCE_loss_ = celoss(batched_output_masks, scribbles[:].long())
        loss = iou_loss_ + dice_loss_ + pCE_loss_
        loss_all += loss
        loss.backward()
        optimizer.step()



    if scheduler:
        scheduler.step()
    avg_loss_all = loss_all / len(data_loader)

    return avg_loss_all


def model_evaluate(
    model,
    data_loader,
    criterion,
    device,
    num 
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model

    Args:
        model (nn.Module): SAM model 
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """

    # Evaluation
    model.eval()
    total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    with torch.no_grad():
        
        running_iouloss = 0.0
        running_diceloss = 0.0
        
        running_dice = 0.0
        running_iou = 0.0
        
        diceloss = criterion[0]        
        iouloss = criterion[1]
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        dice_all = 0.0
        iou_all = 0.0
        len = 0
        for Sample_List in tqdm(data_loader): 
            images, labels, scribbles = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["scribble"].cuda()
            images, labels, scribbles = images.permute(1,0,2,3), labels.permute(1,0,2,3), scribbles.permute(1,0,2,3)
            # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            
            for bs in range(images.shape[0]):
                batched_input_LV, batched_input_MYO, batched_input_RV = [], [], []
                image_bs,label_bs,scribble_bs = images[bs],labels[bs],scribbles[bs]
                for image, label,scribble in zip(image_bs,label_bs, scribble_bs):
                    # prepare image
                    original_size = image.shape[0:3]
                    image = image.unsqueeze(0)
                    image_RGB = torch.cat([image, image, image], dim=0)

                    image_RGB = transform.apply_image(image_RGB)
                    image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
                    image_RGB = image_RGB.permute(2, 0, 1).contiguous()
                    
                    # sampled_point_batch_LV =  max_distance_sample(label.unsqueeze(0),scribble.unsqueeze(0), 1, num)
                    # sampled_point_batch_MYO = max_distance_sample(label.unsqueeze(0),scribble.unsqueeze(0), 2, num)
                    # sampled_point_batch_RV =  max_distance_sample(label.unsqueeze(0),scribble.unsqueeze(0), 3, num)
                    # sampled_point_batch_background = contour_sample_without_bs(scribble, 4, num)
                    
                    sampled_point_batch_LV =  contour_sample_without_bs(scribble, 1, num)
                    sampled_point_batch_MYO = contour_sample_without_bs(scribble, 2, num)
                    sampled_point_batch_RV =  contour_sample_without_bs(scribble, 3, num)
                    sampled_point_batch_background = contour_sample_without_bs(scribble, 4, num)

                    all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO = combine(sampled_point_batch_LV, sampled_point_batch_RV, sampled_point_batch_MYO, sampled_point_batch_background)
                    batched_input_LV, batched_input_MYO, batched_input_RV = process_input_SAM(transform,image_RGB, original_size, all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO, batched_input_LV, batched_input_MYO, batched_input_RV)
                    # plt.figure(figsize=(10,10))
                    # plt.imshow(image[0].cpu().numpy())
                    # plt.imshow(label.cpu().numpy(),alpha=0.5,cmap='gray')
                    # if sampled_point_batch_LV is not None:
                    #     for i in range(sampled_point_batch_LV.shape[0]):
                    #         x, y  = sampled_point_batch_LV[i]
                    #         plt.plot(x, y, 'ro')
                    # plt.axis('off')
                    # plt.savefig(f'data/222/{111}_mask_LV_.png', bbox_inches='tight', pad_inches=0) 
                    # plt.close()

                    # plt.figure(figsize=(10,10))
                    # plt.imshow(image[0].cpu().numpy())
                    # plt.imshow(label.cpu().numpy(),alpha=0.5,cmap='gray')
                    # if sampled_point_batch_RV is not None:
                    #     for i in range(sampled_point_batch_RV.shape[0]):
                    #         x, y  = sampled_point_batch_RV[i]
                    #         plt.plot(x, y, 'ro')

                    # plt.axis('off')
                    # plt.savefig(f'data/222/{111}_mask_RV_.png', bbox_inches='tight', pad_inches=0) 
                    # plt.close()

                    # plt.figure(figsize=(10,10))
                    # plt.imshow(image[0].cpu().numpy())
                    # plt.imshow(label.cpu().numpy(),alpha=0.5,cmap='gray')
                    # if sampled_point_batch_MYO is not None:
                    #     for i in range(sampled_point_batch_MYO.shape[0]):
                    #         x, y  = sampled_point_batch_MYO[i]
                    #         plt.plot(x, y, 'ro')
                    # plt.axis('off')
                    # plt.savefig(f'data/222/{111}_mask_MYO_.png', bbox_inches='tight', pad_inches=0) 
                    # plt.close()

                batched_output_LV = model(batched_input_LV, multimask_output=False)
                batched_output_MYO = model(batched_input_MYO, multimask_output=False)
                batched_output_RV = model(batched_input_RV, multimask_output=False)

                masks_LV =   batched_output_LV[0]['masks_pred'][0][0].cpu().numpy()
                masks_MYO =  batched_output_MYO[0]['masks_pred'][0][0].cpu().numpy()
                masks_RV =   batched_output_RV[0]['masks_pred'][0][0].cpu().numpy()
                mask = np.zeros((masks_LV.shape[0], masks_LV.shape[1]), np.uint8)
                if all_points_MYO is not None:
                    mask[masks_MYO != False] = 2
                if all_points_LV is not None:
                    mask[masks_LV != False]  = 1
                if all_points_RV is not None:
                    mask[masks_RV != False]  = 3
                if all_points_MYO is None and all_points_LV is not None and all_points_RV is not None:
                    mask = np.zeros((masks_LV.shape[0], masks_LV.shape[1]), np.uint8)
                mask = torch.tensor(np.expand_dims(mask, axis = 0)).cuda()
                # plt.figure(figsize=(10,10))
                # plt.imshow(image[0].cpu().numpy())
                # plt.imshow(masks_MYO,alpha=0.5,cmap='gray')
                # if sampled_point_batch_MYO is not None:
                #     for i in range(sampled_point_batch_MYO.shape[0]):
                #         x, y  = sampled_point_batch_MYO[i]
                #         plt.plot(x, y, 'ro')
                # plt.axis('off')
                # plt.savefig(f'data/222/{222}_mask_MYO_.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(image[0].cpu().numpy())
                # plt.imshow(masks_LV,alpha=0.5,cmap='gray')
                # if sampled_point_batch_LV is not None:
                #     for i in range(sampled_point_batch_LV.shape[0]):
                #         x, y  = sampled_point_batch_LV[i]
                #         plt.plot(x, y, 'ro')
                # plt.axis('off')
                # plt.savefig(f'data/222/{222}_mask_LV_.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(image[0].cpu().numpy())
                # plt.imshow(masks_RV,alpha=0.5,cmap='gray')
                # if sampled_point_batch_RV is not None:
                #     for i in range(sampled_point_batch_RV.shape[0]):
                #         x, y  = sampled_point_batch_RV[i]
                #         plt.plot(x, y, 'ro')
                # plt.axis('off')
                # plt.savefig(f'data/222/{222}_mask_RV_.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()


                for j, gt_mask in enumerate(label_bs):
                    
                    if np.unique(gt_mask.cpu().numpy()).any() != 0:
                        gt_mask = gt_mask.unsqueeze(0)

                        dices, HD95,IOU = calculate_metrics(gt_mask, mask, 4)
                        
                            
                        for cls in range(3):
                            total_dice_scores[cls] += dices[cls]
                            total_HD95_scores[cls] += HD95[cls]
                            total_IOU_scores[cls] += IOU[cls]
                        
                    else:
                    ### loss & metrics ###
                        for cls in range(3):
                            total_dice_scores[cls] += 1
                            total_HD95_scores[cls] += 0
                            total_IOU_scores[cls] += 1
                           
                ### update loss & metrics ###

                del image_bs,label_bs,scribble_bs,batched_input_LV, batched_input_MYO, batched_input_RV  # 显式删除无用变量
                torch.cuda.empty_cache()

            len += images.shape[0]

        
        ### Average loss & metrics ### 
        
        avg_dice_scores = [(total / len).cpu().numpy() for total in total_dice_scores]
        avg_HD95_scores = [(total / len) for total in total_HD95_scores]
        avg_IOU_scores =  [(total / len).cpu().numpy() for total in total_IOU_scores]
        
        avg_dice = np.mean(avg_dice_scores)
        avg_HD95 = np.mean(avg_HD95_scores)
        avg_IOU =  np.mean(avg_IOU_scores)


    return avg_dice, avg_HD95,avg_IOU

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', verbose=True):
        """
        Pytorch Early Stopping

        Args:
            patience (int, optional): patience. Defaults to 10.
            delta (float, optional): threshold to update best score. Defaults to 0.0.
            mode (str, optional): 'min' or 'max'. Defaults to 'min'(comparing loss -> lower is better).
            verbose (bool, optional): verbose. Defaults to True.
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        
    def __call__(self, score):
        # self.best_score = 15.0
        # _1 = np.abs(self.best_score - score.cpu())
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}')
                    # , Delta: {np.abs(self.best_score - score.cpu().numpy()):.5f}
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}')
                
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False

def model_train_cell(
    model,
    data_loader,
    criterion,
    optimizer,        
    device,
    scheduler,
    num
) -> Tuple[float, float, float, float]:
    """
    Train the model

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions 
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """
    
    # Training
    model.train()
    
    running_iouloss = 0.0
    running_diceloss = 0.0
    
    running_dice = 0.0
    running_iou = 0.0
    
    n_data = 0
    
    diceloss = criterion[0]    
    iouloss = criterion[1]
    celoss = criterion[2]
    
    transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
    loss_all = 0.0
    for Sample_List in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels, PLs, scribbles, id = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["pesudo_label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
        
        batched_input_Nu, batched_input_Cy = [], []
        for image, scribble in zip(images, scribbles):
            original_size = image.shape[1:3]
            image_RGB = transform.apply_image(image)
 
            image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
            image_RGB = image_RGB.permute(2, 0, 1).contiguous()
            
            sampled_point_batch_Nu =  contour_sample_without_bs(scribble, 1, num)
            sampled_point_batch_Cy = contour_sample_without_bs(scribble, 2, num)
            sampled_point_batch_background = contour_sample_without_bs(scribble, 0, num)

            all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy = combine_cell(sampled_point_batch_Nu, sampled_point_batch_Cy, sampled_point_batch_background)
            batched_input_Nu, batched_input_Cy = process_input_SAM_cell(transform,image_RGB, original_size,all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy, batched_input_Nu, batched_input_Cy)
            
        batched_inputs = [batched_input_Nu, batched_input_Cy]
        batched_output_masks,batched_output_masks_pred = -1e5*torch.zeros((images.shape[0],3,images.shape[2],images.shape[3])).cuda(),-1e5*torch.zeros((images.shape[0],3,images.shape[2],images.shape[3])).cuda()
        masks = torch.zeros((images.shape[2],images.shape[3])).cuda()
        masks_pred = torch.zeros((images.shape[2],images.shape[3])).cuda()
        for i, batched_input_cls in enumerate(batched_inputs):

            batched_output = model(batched_input_cls, multimask_output=False)
            
            for j in range(data_loader.batch_size):
                batched_output_masks[j,i+1,:,:] = batched_output[j]["masks"][0][0]
                batched_output_masks_pred[j,i+1,:,:] = batched_output[j]["masks_pred"][0][0]
                # masks += batched_output_masks[j,i+1,:,:]  
        batched_output_masks[:,0,:,:] = torch.mean(batched_output_masks, dim=1)
        batched_output_masks_pred[:,0,:,:] = torch.mean(batched_output_masks_pred, dim=1)
        batched_output_masks_pred_mask = torch.argmax(batched_output_masks_pred,dim= 1 )
        
        iou_loss_ = iouloss(batched_output_masks_pred, PLs.unsqueeze(1))
        dice_loss_ = diceloss(batched_output_masks_pred, PLs.unsqueeze(1)) 
        pCE_loss_ = celoss(batched_output_masks, scribbles[:].long())
        loss = iou_loss_ + dice_loss_ + pCE_loss_
        loss_all += loss
        loss.backward()
        optimizer.step()  
        
        
    if scheduler:
        scheduler.step()
    avg_loss_all = loss_all / len(data_loader)

    return avg_loss_all


def model_evaluate_cell(
    model,
    data_loader,
    criterion,
    device,
    num 
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model

    Args:
        model (nn.Module): SAM model 
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """

    # Evaluation
    model.eval()
    total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    # len_data = len(data_loader.dataset)
    with torch.no_grad():
        
        running_iouloss = 0.0
        running_diceloss = 0.0
        
        running_dice = 0.0
        running_iou = 0.0
        
        diceloss = criterion[0]    
        iouloss = criterion[1]
        
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        len_data = 0 
        
        for Sample_List in tqdm(data_loader): 
            images, labels, scribbles,id  = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
            batched_input_Nu, batched_input_Cy = [], []
            for image, scribble in zip(images, scribbles):
                
                original_size = image.shape[1:3]
                
                image_RGB = transform.apply_image(image) 
                image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
                image_RGB = image_RGB.permute(2, 0, 1).contiguous()
                

                sampled_point_batch_Nu =  contour_sample_without_bs(scribble, 1, num)
                sampled_point_batch_Cy = contour_sample_without_bs(scribble, 2, num)
                sampled_point_batch_background = contour_sample_without_bs(scribble, 0, num)
                all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy = combine_cell(sampled_point_batch_Nu, sampled_point_batch_Cy, sampled_point_batch_background)
                batched_input_Nu, batched_input_Cy = process_input_SAM_cell(transform,image_RGB, original_size,all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy, batched_input_Nu, batched_input_Cy)

            batched_output_Nu = model(batched_input_Nu, multimask_output=False)
            batched_output_Cy = model(batched_input_Cy, multimask_output=False)

            masks_Nu =   batched_output_Nu[0]['masks_pred'][0][0].cpu().numpy()
            masks_Cy =  batched_output_Cy[0]['masks_pred'][0][0].cpu().numpy()
            
            mask = np.zeros((masks_Nu.shape[0], masks_Nu.shape[1]), np.uint8)
            if all_points_Cy is not None:
                mask[masks_Cy != False] = 2
            if all_points_Nu is not None:
                mask[masks_Nu != False]  = 1
            if all_points_Nu is None and all_points_Cy is None:
                mask = np.zeros((masks_Nu.shape[0], masks_Nu.shape[1]), np.uint8)

            mask = torch.tensor(np.expand_dims(mask, axis = 0)).cuda()

            dice = 0.0
            iou = 0.0
            
            for j, gt_mask in enumerate(labels):
                if np.unique(gt_mask.cpu().numpy()).any() != 0:
                    gt_mask = gt_mask.unsqueeze(0)

                    
                    dices, HD95,IOU = calculate_metrics(gt_mask, mask, 3)
                            
                    for cls in range(2):
                        total_dice_scores[cls] += dices[cls]
                        total_HD95_scores[cls] += HD95[cls]
                        total_IOU_scores[cls] += IOU[cls]
                        
                else:
                ### loss & metrics ###
                    for cls in range(2):
                        total_dice_scores[cls] += 1
                        total_HD95_scores[cls] += 0
                        total_IOU_scores[cls] += 1

            # torch.cuda.empty_cache()

            len_data += images.shape[0]
        avg_dice_scores = [(total / len_data).cpu().numpy() for total in total_dice_scores]
        avg_HD95_scores = [(total / len_data) for total in total_HD95_scores]
        avg_IOU_scores =  [(total / len_data).cpu().numpy() for total in total_IOU_scores]
        
        avg_dice = np.mean(avg_dice_scores)
        avg_HD95 = np.mean(avg_HD95_scores)
        avg_IOU =  np.mean(avg_IOU_scores)

        print(f'category LV:   Dice: {avg_dice_scores[0]:.3f},  HD95: {avg_HD95_scores[0]:.3f},  IOU: {avg_IOU_scores[0]:.3f}')
        print(f'category MYO:  Dice: {avg_dice_scores[1]:.3f},  HD95: {avg_HD95_scores[1]:.3f},  IOU: {avg_IOU_scores[1]:.3f}')
        print(f'Total:       Dice: {avg_dice:.3f},  HD95: {avg_HD95:.3f},  IoU: {avg_IOU:.3f}')

    return avg_dice, avg_HD95,avg_IOU



def model_train_choas(
    model,
    data_loader,
    criterion,
    optimizer,        
    device,
    scheduler,
    num
) -> Tuple[float, float, float, float]:
    """
    Train the model

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions 
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """
    
    # Training
    model.train()
    
    running_iouloss = 0.0
    running_diceloss = 0.0
    
    running_dice = 0.0
    running_iou = 0.0
    
    n_data = 0
    
    diceloss = criterion[0]    
    iouloss = criterion[1]
    celoss = criterion[2]

    transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
    loss_all = 0.0
    for Sample_List in tqdm(data_loader):
        optimizer.zero_grad()
        images, labels, PLs, scribbles, id = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["pesudo_label"].cuda(), Sample_List["scribble"].cuda(),Sample_List["idx"]
        labels_np = np.array(labels.cpu())
        
        # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
        
        batched_input_LIV, batched_input_RK, batched_input_LK, batched_input_LSP= [], [], [],[]
        for image, label,PL, scribble in zip(images,labels,PLs, scribbles):
            # prepare image
            original_size = image.shape[1:3]
            image_RGB = torch.cat([image, image, image], dim=0)

            image_RGB = transform.apply_image(image_RGB)
            image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
            image_RGB = image_RGB.permute(2, 0, 1).contiguous()

            sampled_point_batch_LIV =  contour_sample_without_bs(scribble, 1, num)
            sampled_point_batch_RK = contour_sample_without_bs(scribble, 2, num)
            sampled_point_batch_LK =  contour_sample_without_bs(scribble, 3, num)
            sampled_point_batch_LSP =  contour_sample_without_bs(scribble, 4, num)
            sampled_point_batch_background = contour_sample_without_bs(scribble, 5, num)


            all_points_LIV, all_labels_LIV, all_points_RK,  all_labels_RK, all_points_LK, all_labels_LK,all_points_LSP,  all_labels_LSP = combine_choas(sampled_point_batch_LIV, sampled_point_batch_RK, sampled_point_batch_LK, sampled_point_batch_LSP,sampled_point_batch_background)
            
            batched_input_LIV, batched_input_RK, batched_input_LK,batched_input_LSP = process_input_SAM_Choas(transform,image_RGB,
             original_size, all_points_LIV, all_labels_LIV, all_points_RK,  all_labels_RK, all_points_LK, all_labels_LK,
             all_points_LSP,  all_labels_LSP, batched_input_LIV, batched_input_RK, batched_input_LK,batched_input_LSP)
            
            
        batched_inputs = [batched_input_LIV, batched_input_RK, batched_input_LK,batched_input_LSP]
        batched_output_masks,batched_output_masks_pred = -1e5*torch.zeros((images.shape[0],5,images.shape[2],images.shape[3])).cuda(),-1e5*torch.zeros((images.shape[0],5,images.shape[2],images.shape[3])).cuda()
        masks = torch.zeros((images.shape[2],images.shape[3])).cuda()
        masks_pred = torch.zeros((images.shape[2],images.shape[3])).cuda()
        for i, batched_input_cls in enumerate(batched_inputs):
            batched_output = model(batched_input_cls, multimask_output=False)
            for j in range(data_loader.batch_size):
                a = batched_input_cls[j] 
                if 'point_coords' in batched_input_cls[j]:
                    batched_output_masks[j,i+1,:,:] = batched_output[j]["masks"][0][0]
                    batched_output_masks_pred[j,i+1,:,:] = batched_output[j]["masks_pred"][0][0]
                # plt.figure(figsize=(10,10))
                # plt.imshow(images[j][0].cpu().numpy())
                # plt.imshow(batched_output_masks_pred[j,i+1,:,:].cpu().detach().numpy(),alpha=0.5,cmap='gray')
                # plt.axis('off')
                # plt.savefig(f'data/222/{j}_{i+1}_pred.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(images[j][0].cpu().numpy())
                # plt.imshow((labels[j]==i+1).cpu().detach().numpy(),alpha=0.5,cmap='gray')
                # plt.axis('off')
                # plt.savefig(f'data/222/{j}_{i+1}_mask.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()
            batched_output_masks[:,0,:,:] = torch.mean(batched_output_masks, dim=1)
            batched_output_masks_pred[:,0,:,:] = torch.mean(batched_output_masks_pred, dim=1)
            # batched_output_masks_pred_mask = torch.argmax(batched_output_masks_pred,dim= 1 )

        iou_loss_ = iouloss(batched_output_masks, PLs.unsqueeze(1))
        dice_loss_ = diceloss(batched_output_masks, PLs.unsqueeze(1)) 
        pCE_loss_ = celoss(batched_output_masks, scribbles[:].long())
        loss = iou_loss_ + dice_loss_ + pCE_loss_
        loss_all += loss
        loss.backward()
        optimizer.step()

    



    if scheduler:
        scheduler.step()
    avg_loss_all = loss_all / len(data_loader)

    return avg_loss_all


def model_evaluate_choas(
    model,
    data_loader,
    criterion,
    device,
    num 
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model

    Args:
        model (nn.Module): SAM model 
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)

    Returns:
        Tuple[float, float, float, float]: average losses(dice, iou), metrics(dice, iou)
    """

    # Evaluation
    model.eval()
    total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0, 0.0,0.0],[0.0, 0.0, 0.0,0.0],[0.0, 0.0, 0.0,0.0]    # len_data = len(data_loader.len())
    with torch.no_grad():
        
        running_iouloss = 0.0
        running_diceloss = 0.0
        
        running_dice = 0.0
        running_iou = 0.0
        
        diceloss = criterion[0]        
        iouloss = criterion[1]
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        dice_all = 0.0
        iou_all = 0.0
        len = 0
        for Sample_List in tqdm(data_loader): 
            images, labels, scribbles = Sample_List["image"].cuda(),  Sample_List["label"].cuda(), Sample_List["scribble"].cuda()
            images, labels, scribbles = images.permute(1,0,2,3), labels.permute(1,0,2,3), scribbles.permute(1,0,2,3)
            # X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            
            for bs in range(images.shape[0]):
                batched_input_LIV, batched_input_RK, batched_input_LK, batched_input_LSP= [], [], [],[]
                image_bs,label_bs,scribble_bs = images[bs],labels[bs],scribbles[bs]
                for image, label,scribble in zip(image_bs,label_bs, scribble_bs):
                    # prepare image
                    original_size = image.shape[0:3]
                    image = image.unsqueeze(0)
                    image_RGB = torch.cat([image, image, image], dim=0)

                    image_RGB = transform.apply_image(image_RGB)
                    image_RGB = torch.as_tensor(image_RGB, dtype=torch.float, device=device)
                    image_RGB = image_RGB.permute(2, 0, 1).contiguous()
                  
                    sampled_point_batch_LIV =  contour_sample_without_bs(scribble, 1, num)
                    sampled_point_batch_RK = contour_sample_without_bs(scribble, 2, num)
                    sampled_point_batch_LK =  contour_sample_without_bs(scribble, 3, num)
                    sampled_point_batch_LSP =  contour_sample_without_bs(scribble, 4, num)
                    sampled_point_batch_background = contour_sample_without_bs(scribble, 5, num)

                    all_points_LIV, all_labels_LIV, all_points_RK,  all_labels_RK, all_points_LK, all_labels_LK,all_points_LSP,  all_labels_LSP = combine_choas(sampled_point_batch_LIV, sampled_point_batch_RK, sampled_point_batch_LK, sampled_point_batch_LSP,sampled_point_batch_background)
            
                    batched_input_LIV, batched_input_RK, batched_input_LK,batched_input_LSP = process_input_SAM_Choas(transform,image_RGB,
                    original_size, all_points_LIV, all_labels_LIV, all_points_RK,  all_labels_RK, all_points_LK, all_labels_LK,
                    all_points_LSP,  all_labels_LSP, batched_input_LIV, batched_input_RK, batched_input_LK,batched_input_LSP)


                batched_output_LIV = model(batched_input_LIV, multimask_output=False)
                batched_output_RK = model(batched_input_RK, multimask_output=False)
                batched_output_LK = model(batched_input_LK, multimask_output=False)
                batched_output_LSP = model(batched_input_LSP, multimask_output=False)

                masks_LIV =   batched_output_LIV[0]['masks_pred'][0][0].cpu().numpy()
                masks_RK =  batched_output_RK[0]['masks_pred'][0][0].cpu().numpy()
                masks_LK =   batched_output_LK[0]['masks_pred'][0][0].cpu().numpy()
                masks_LSP =   batched_output_LSP[0]['masks_pred'][0][0].cpu().numpy()
                mask = np.zeros((masks_LIV.shape[0], masks_LIV.shape[1]), np.uint8)
                if all_points_LSP is not None:
                    mask[masks_LSP[0] != False] = 4
                if all_points_LK is not None:
                    mask[masks_LK[0] != False]  = 3
                if all_points_LIV is not None:
                    mask[masks_LIV[0] != False] = 1
                if all_points_RK is not None:
                    mask[masks_RK[0] != False]  = 2
                mask = torch.tensor(np.expand_dims(mask, axis = 0)).cuda()
                


                for j, gt_mask in enumerate(label_bs):
                    
                    if np.unique(gt_mask.cpu().numpy()).any() != 0:
                        gt_mask = gt_mask.unsqueeze(0)

                        dices, HD95,IOU = calculate_metrics(gt_mask, mask, 5)
                        
                            
                        for cls in range(4):
                            total_dice_scores[cls] += dices[cls]
                            total_HD95_scores[cls] += HD95[cls]
                            total_IOU_scores[cls] += IOU[cls]
                        
                    else:

                        for cls in range(4):
                            total_dice_scores[cls] += 1
                            total_HD95_scores[cls] += 0
                            total_IOU_scores[cls] += 1


            
                torch.cuda.empty_cache()

            len += images.shape[0]


        
        avg_dice_scores = [(total / len).cpu().numpy() for total in total_dice_scores]
        avg_HD95_scores = [(total / len) for total in total_HD95_scores]
        avg_IOU_scores =  [(total / len).cpu().numpy() for total in total_IOU_scores]
        
        avg_dice = np.mean(avg_dice_scores)
        avg_HD95 = np.mean(avg_HD95_scores)
        avg_IOU =  np.mean(avg_IOU_scores)


    return avg_dice, avg_HD95,avg_IOU
