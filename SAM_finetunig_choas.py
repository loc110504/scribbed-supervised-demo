import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import logging
import os
import argparse
import numpy as np 
from losses import SAM_loss
from segment_anything import sam_model_registry
from my_utils import SAM_trainer,save_weight
from dataset.dataset_Choas import BaseDataSets_SAM,RandomGenerator_SAM
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
import random
from PIL import Image
from my_utils.metrics import calculate_metrics
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from my_utils.Sampling_Combine import random_sample, contour_sample, combine, Entropy_Grids_Sampling,Entropy_contour_Sampling,contour_sample_without_bs,process_input_SAM


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root_path', type=str, default='/home/cj/code/model_New/datasets/CHoas/T2')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size allocated to each GPU')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--num', type=int, default=10)
    
    parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default='/home/cj/code/model_New/SAM_FineTune/sam_vit_h_4b8939.pth', help='SAM model checkpoint')
    parser.add_argument('--exp', type=str, default='SAM_FineTune/iteration1_vith/ChoasT1')
    parser.add_argument('--iter', type=str, default='iteration2')
    
    parser.add_argument('--max_epochs', type=int, default=30, help='total epoch')
    parser.add_argument('--base_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--deterministic', type=int,  default=1,help='whether use deterministic training')
    parser.add_argument('--fold', type=str,  default="fold5",help='whether use deterministic training')
    return parser

def main(opts, snapshot_path, savepath):
    
    num_classes = opts.num_classes
    max_iterations = opts.max_epochs
    bestval = 0
    ### Dataset & Dataloader ### 
    train_set = BaseDataSets_SAM(base_dir=opts.root_path, fold=opts.fold, transform=transforms.Compose(
        [RandomGenerator_SAM(opts.patch_size,split='train')]), split="train",pesudo_label = 'SAM_PL')
    

    val_set = BaseDataSets_SAM(base_dir=opts.root_path, fold=opts.fold, split="val", transform=transforms.Compose(
        [RandomGenerator_SAM(opts.patch_size,split='val')]),pesudo_label = 'SAM_PL')

    train_loader = DataLoader(
        train_set, 
        batch_size=opts.batch_size, 
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        shuffle=False,
        drop_last=True
    )
    
    ### Model config ### 
    
    sam_checkpoint = opts.sam_checkpoint
    model_type = opts.sam_model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)


    sam.cuda()

    # set trainable parameters
    for _, p in sam.image_encoder.named_parameters():
        p.requires_grad = False
        
    for _, p in sam.prompt_encoder.named_parameters():
        p.requires_grad = False

    # fine-tuning mask decoder         
    for _, p in sam.mask_decoder.named_parameters():
        p.requires_grad = True
        
        
    ### Training config ###  
   
    iouloss = SAM_loss.IoULoss()
    diceloss = SAM_loss.pDLoss(5,ignore_index=4)
    ce_loss = CrossEntropyLoss(ignore_index=5)

    es = SAM_trainer.EarlyStopping(patience=30, delta=0, mode='min', verbose=True)
    optimizer = torch.optim.AdamW(sam.parameters(), lr=opts.base_lr, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )
    
    max_loss = np.inf
    best_dice = 0.0
    ### Training phase ###
    
    for epoch in range(opts.max_epochs):
        print(f'# Epochs {epoch}')
        train_dice_loss = SAM_trainer.model_train_choas(
            model=sam,
            data_loader=train_loader,
            criterion=[diceloss, iouloss,ce_loss],
            optimizer=optimizer,
            device='cuda',
            scheduler=scheduler,
            num = opts.num
        )

        val_dice, val_HD95, val_iou = SAM_trainer.model_evaluate_choas(
            model=sam,
            data_loader=val_loader,
            criterion=[diceloss, iouloss],
            device='cuda',
            num = opts.num
        )
        

        save_best_path = os.path.join(savepath,'sam_best_decoder.pth')
        save_best_path_all_perepoch = os.path.join(savepath,f'sam_all_{epoch}_{val_dice}.pth')
        save_best_path_all = os.path.join(savepath,f'sam_best_all.pth')
        # save best model 
        if val_dice > best_dice:
            print(f'[INFO] val_dice has been improved from {best_dice:.5f} to {val_dice:.5f}. Save model.')
            best_dice = val_dice
            _ = save_weight.save_partial_weight(model=sam, save_path=save_best_path)
            torch.save(sam.state_dict(), save_best_path_all)
            torch.save(sam.state_dict(), save_best_path_all_perepoch)
            print("save model to {}".format(save_best_path_all))
        if epoch%3 ==0:
            torch.save(sam.state_dict(), save_best_path_all_perepoch)
        # print current loss & metric
        print(f'epoch {epoch+1:02d}, dice_loss: {train_dice_loss:.5f} \n')
        print(f'val_dice: {val_dice:.5f}, val_iou: {val_iou:.5f}, val_HD95:{val_HD95:.5f} \n')
        
        if es.early_stop:
            break    

    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Iterative Re-Training', parents=[get_args_parser()])
    opts = parser.parse_args()

    if not opts.deterministic:
        opts.benchmark = True
        opts.deterministic = False
    else:
        opts.benchmark = False
        opts.deterministic = True
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)\
    
    savepath =  "../model_New/{}".format(
        opts.exp)
    snapshot_path = "../model_New/{}".format(
        opts.exp,)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # model_type =os.path.split(opts.checkpoint)[1][4:9]
    # sam_model = sam_model_registry[model_type](opts.checkpoint).cuda()
    # predictor = SamPredictor(sam_model)
    # dataset = BaseDataSets_SAM(base_dir="/home/lxy/pycharm_project/SAM_Scribble/data/ACDC", transform=transforms.Compose([
    #                             RandomGenerator_SAM([256, 256])]
    #                             ),split="train",  fold='fold1', sup_type='scribble')
    # dataloader = DataLoader(dataset, 1 , shuffle= False)
    logging.info(str(opts))

    print('=== Iterative Re-Training ===')
    
    print(f'# Iteration {opts.iter}')
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    main(opts, snapshot_path, savepath)
    
    print('=== DONE === \n')    