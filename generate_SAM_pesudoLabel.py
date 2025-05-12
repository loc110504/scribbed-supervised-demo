import argparse
import logging
import os
import random
import shutil
import sys
import time
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
# from losses.abl_loss import ABL
from losses.pd_loss import pDLoss
from matplotlib import pyplot as plt

# from dataset.dataset_ACDC import BaseDataSets_Net_G_SAM_PL, RandomGenerator_Net_G_SAM_PL
from dataset.dataset_MSCMR import BaseDataSets_Net_G_SAM_PL, RandomGenerator_Net_G_SAM_PL
from Networks.net_factory import net_factory
from my_utils.val2D import test_single_volume_cct, test_single_volume_ds
from PIL import Image 

dataset_name = "MSCMR"
model_name = "vit_h"
root_path='/home/cj/code/model_New/datasets/MSCMR'
model = "/home/cj/code/model_New/comparison/MSCMR/vit_h/fold5"
net_type = 'unet_DMPLS'

# model = "/home/cj/code/model_New/test/MSCMR/iteration2/3/fold1"
# net_type = 'unet_DMPLS_att'

savepath = "/home/cj/code/model_New/comparison/vis"


# modelpath = os.path.join(model, "scribble/"+net_type+"_best_model.pth")
modelpath = "/home/cj/code/model_New/comparison/MSCMR/MedSAM/fold1/scribble/iter_500_dice_0.7953.pth"


vis_path = os.path.join(savepath, f"{dataset_name}/{model_name}/result")
result_path = os.path.join(savepath, f"{dataset_name}/{model_name}/pl")
os.makedirs(vis_path,exist_ok=True)
os.makedirs(result_path,exist_ok=True)
patch_size = [256,256]

num_classes = 4

model = net_factory(net_type=net_type, in_chns=1, class_num=num_classes)
# 补足模型路径
model_weight = torch.load(modelpath)
model.load_state_dict(model_weight)
db_val = BaseDataSets_Net_G_SAM_PL(base_dir=root_path,transform=transforms.Compose([
    RandomGenerator_Net_G_SAM_PL(patch_size)
]))
valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                        num_workers=0)
model.eval() 
for i_batch, sampled_batch in enumerate(tqdm(valloader)):
    image, id = sampled_batch['image'].unsqueeze(0).cuda(), sampled_batch['idx'][0]
    label = sampled_batch['label'].numpy()
    id = os.path.splitext(id)[0]
    output_main = model(image)[0]
    
    out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
    
    out = out.detach().cpu().numpy().astype(np.uint8())


    cmap1 = ListedColormap(["black", "red", "blue","green"])
    plt.figure(figsize=(10,10))
    plt.imshow(image[0][0].cpu().numpy(),cmap='gray')
    # plt.imshow(out, cmap=cmap1)
    plt.imshow(out, cmap=cmap1,alpha=0.5)      
     
    plt.axis('off')
    plt.savefig(f'{vis_path}/{id}.png', bbox_inches='tight', pad_inches=0)
    plt.close()





    






    
    

