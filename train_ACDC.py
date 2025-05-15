import argparse
import logging
import os
import random
import shutil
import sys
import time

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
import torch_scatter
# from losses.abl_loss import ABL
from losses.pd_loss import pDLoss
from matplotlib import pyplot as plt
from dataset.dataset_ACDC import BaseDataSets, RandomGenerator
from Networks.net_factory import net_factory
# from utils import losses, metrics, ramps
# from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from my_utils.val2D import test_single_volume_cct, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='XXX', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/InterFA_SA_Two', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_DMPLS_att', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--T', type=int,  default=1, help='SSA T')
parser.add_argument('--PL_path', type=str,
                    default='ACDC_training_conf_iteration2', help='Name of Experiment')
parser.add_argument('--ca_iteration', type=int, default=10000)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length


def train(args, snapshot_path, savepath):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size,split='train')
    ]), fold=args.fold, sup_type=args.sup_type,pseudo_label_path=args.PL_path)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold,transform=transforms.Compose([
        RandomGenerator(args.patch_size,split='val')
    ]), split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0,drop_last=True)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = pDLoss(num_classes, ignore_index=4)


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, pseudo_label_batch,conf = sampled_batch['image'].cuda(), sampled_batch['scribble'].cuda(), sampled_batch['SAM_PL'].cuda(),sampled_batch['conf'].cuda()

######################################################################
###########################  Loss Function ###########################
######################################################################
            # mere pCE Loss

            outputs, mainseg, auxseg, latent_feature = model(volume_batch)
            outputs_soft= torch.softmax(outputs, dim=1)
            outputs_soft_1= torch.softmax(mainseg, dim=1)
            outputs_soft_2= torch.softmax(auxseg, dim=1)

            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2= ce_loss(mainseg, label_batch[:].long())
            loss_ce3= ce_loss(auxseg, label_batch[:].long())

            loss_ce = loss_ce1 + loss_ce2 + loss_ce3


            latent_feature = F.normalize(latent_feature, dim=1)
            latent_feature = latent_feature.view(args.batch_size,256,-1)
            index_ = pseudo_label_batch.view(args.batch_size,1,-1).long()

            pt = torch_scatter.scatter_mean(latent_feature.detach(), index_)
            pt = F.normalize(pt, dim=1)
            index_ = index_.squeeze(1)
            pred_ssa = torch.bmm(pt.permute(0,2,1),latent_feature)

            loss_ssa_item = F.cross_entropy(pred_ssa * args.T, index_, ignore_index=0)
            if not torch.isnan(loss_ssa_item):
                loss_ssa = loss_ssa_item
            else:
                print("loss_ssa is NaN!")
                loss_ssa = 0



            # SAM-based Confidence Aggregation Loss
            if iter_num>=args.ca_iteration:
                sam_conf = -1e5*torch.ones((outputs.shape[0],outputs.shape[1],outputs.shape[2],outputs.shape[3])).cuda()
                conf_bg = torch.zeros((outputs.shape[2],outputs.shape[3])).cuda()

                sam_conf_1 = -1e5*torch.ones((outputs.shape[0],outputs.shape[1],outputs.shape[2],outputs.shape[3])).cuda()
                conf_bg_1 = torch.zeros((outputs.shape[2],outputs.shape[3])).cuda()

                sam_conf_2 = -1e5*torch.ones((outputs.shape[0],outputs.shape[1],outputs.shape[2],outputs.shape[3])).cuda()
                conf_bg_2 = torch.zeros((outputs.shape[2],outputs.shape[3])).cuda()
                for i in range(batch_size):
                    for k in range(1,(num_classes)):
                        area = pseudo_label_batch[i]==k

                        sam_conf[i,k][area] = conf[i,k-1][area] * outputs_soft[i,k][area].mean()
                        conf_bg += conf[i,k-1]

                        sam_conf_1[i,k][area] = conf[i,k-1][area] * outputs_soft_1[i,k][area].mean()
                        conf_bg_1 += conf[i,k-1]

                        sam_conf_2[i,k][area] = conf[i,k-1][area] * outputs_soft_2[i,k][area].mean()
                        conf_bg_2 += conf[i,k-1]

                    area = pseudo_label_batch[i]==0

                    conf_bg = conf_bg/3
                    sam_conf[i,0][area] = conf_bg[area]*outputs_soft[i,0][area].mean()

                    conf_bg_1 = conf_bg_1/3
                    sam_conf_1[i,0][area] = conf_bg_1[area]*outputs_soft_1[i,0][area].mean()

                    conf_bg_2 = conf_bg_2/3
                    sam_conf_2[i,0][area] = conf_bg_2[area]*outputs_soft_2[i,0][area].mean()

                temp = sam_conf.max(dim=1)
                pgt_sam = temp[1]
                pgt_score = temp[0]
                pgt_sam[pgt_score<0] = 4
                pgt_score[pgt_score<0] = 0


                temp_1 = sam_conf_1.max(dim=1)
                pgt_sam_1 = temp_1[1]
                pgt_score_1 = temp_1[0]
                pgt_sam_1[pgt_score_1<0] = 4
                pgt_score_1[pgt_score_1<0] = 0


                temp_2 = sam_conf_2.max(dim=1)
                pgt_sam_2 = temp_2[1]
                pgt_score_2 = temp_2[0]
        

                pgt_sam_2[pgt_score_2<0] = 4
                pgt_score_2[pgt_score_2<0] = 0
                
                # loss_sca = dice_loss(outputs_soft, pgt_sam.unsqueeze(1))
                loss_sca = F.cross_entropy(outputs, pgt_sam,ignore_index=4) + F.cross_entropy(outputs, pgt_sam_1,ignore_index=4) + F.cross_entropy(outputs, pgt_sam_2,ignore_index=4)

                if not torch.isnan(loss_sca):
                    loss_SCA = loss_sca
                else:
                    print("loss_SCA is NaN!")
                    loss_SCA = torch.zeros_like(loss_ce)


            else:
                loss_SCA = dice_loss(outputs_soft, pseudo_label_batch.unsqueeze(1)) + dice_loss(outputs_soft_2, pseudo_label_batch.unsqueeze(1)) +dice_loss(outputs_soft_2, pseudo_label_batch.unsqueeze(1)) 


            loss = loss_ce + loss_SCA + loss_ssa

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_SCA: %f, alpha: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_SCA.item(), alpha))

            if iter_num % 200 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_cct(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                dice_cls0,dice_cls1,dice_cls2 = metric_list[0,0],metric_list[1,0],metric_list[2,0]
                HD95_cls0,HD95_cls1,HD95_cls2 = metric_list[0,1],metric_list[1,1],metric_list[2,1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance,best_hd95 = performance,mean_hd95
                    best_dice_0,best_dice_1,best_dice_2 = dice_cls0,dice_cls1,dice_cls2
                    best_hd95_0,best_hd95_1,best_hd95_2 = HD95_cls0,HD95_cls1,HD95_cls2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                logging.info('iteration %d : LV_dice : %f LV_hd95 : %f' % (iter_num, dice_cls0, HD95_cls0))
                logging.info('iteration %d : MYO_dice : %f MYO_hd95 : %f' % (iter_num, dice_cls1, HD95_cls1))
                logging.info('iteration %d : RV_dice : %f RV_hd95 : %f' % (iter_num, dice_cls2, HD95_cls2))
                    
                
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    import csv
    with open(os.path.join(savepath, 'best_metrics.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['','mean_dice','mean_hd95', 'cls0_dice','cls0_hd95', 'cls1_dice','cls1_hd95', 'cls2_dice','cls2_hd95'])
        writer.writerow([args.fold, best_performance,best_hd95,best_dice_0,best_hd95_0,best_dice_1,best_hd95_1,best_dice_2,best_hd95_2])   
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    savepath =  "../model_New/{}".format(
        args.exp)
    snapshot_path = "../model_New/{}/{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path,savepath)
