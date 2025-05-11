import argparse
import logging
import os
import random
import shutil
import sys
import time
import torch_scatter
import numpy as np
import torch
# from matplotlib import pyplot as plt
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
from medpy import metric
from dataset.dataset_SegPC import BaseDataSets_cell, RandomGenerator_cell
from Networks.net_factory import net_factory
# from utils import losses, metrics, ramps
# from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from my_utils.val2D import test_single_volume_cct


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='xxx', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='cell/InterFA_SA_Two_thin_aux_iteartion', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_DMPLS_att', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=9000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--ca_iteration', type=int, default=3000)
parser.add_argument('--T', type=int,  default=1, help='SSA T')
args = parser.parse_args()


def calculate_metrics(y_true, y_pred, num_classes):
    dices = []
    hd95 = []
    for cls in range(1,num_classes):
        cls_y_true = (y_true == cls).cpu().detach().numpy()
        cls_y_pred = (y_pred == cls).cpu().detach().numpy()
        cls_y_true[cls_y_true > 0] = 1
        cls_y_pred[cls_y_pred > 0] = 1  
        if cls_y_pred.sum() > 0:
            dices.append(metric.binary.dc(cls_y_true, cls_y_pred))
            hd95.append(metric.binary.hd95(cls_y_true, cls_y_pred))
        else:
            dices.append(1.0)
            hd95.append(0.0)


    return dices, hd95


def train(args, snapshot_path, savepath):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
    db_train = BaseDataSets_cell(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator_cell(args.patch_size,split='train')
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets_cell(base_dir=args.root_path, fold=args.fold,transform=transforms.Compose([
        RandomGenerator_cell(args.patch_size,split='val')
    ]), split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=base_lr,weight_decay=1e-2)
    ce_loss = CrossEntropyLoss(ignore_index=3)
    dice_loss = pDLoss(num_classes, ignore_index=3)


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, pesudo_label_batch,conf = sampled_batch['image'].cuda(), sampled_batch['scribble'].cuda(), sampled_batch['pesudo_label'].cuda(),sampled_batch['conf'].cuda()
        
            outputs, mainseg, auxseg, latent_feature = model(volume_batch)
            outputs_soft= torch.softmax(outputs, dim=1)
            outputs_soft_1= torch.softmax(mainseg, dim=1)
            outputs_soft_2= torch.softmax(auxseg, dim=1)

            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2= ce_loss(mainseg, label_batch[:].long())
            loss_ce3= ce_loss(auxseg, label_batch[:].long())

            loss_ce = loss_ce1 + loss_ce2 + loss_ce3


            # SAM-Semantic Alignment Loss
            latent_feature = F.normalize(latent_feature, dim=1)
            latent_feature = latent_feature.view(args.batch_size,256,-1)
            index_ = pesudo_label_batch.view(args.batch_size,1,-1).long()

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
                for i in range(outputs.shape[0]):
                    for k in range(1,(num_classes)):
                        area = pesudo_label_batch[i]==k

                        sam_conf[i,k][area] = conf[i,k-1][area] * outputs_soft[i,k][area].mean()
                        conf_bg += conf[i,k-1]

                        sam_conf_1[i,k][area] = conf[i,k-1][area] * outputs_soft_1[i,k][area].mean()
                        conf_bg_1 += conf[i,k-1]

                        sam_conf_2[i,k][area] = conf[i,k-1][area] * outputs_soft_2[i,k][area].mean()
                        conf_bg_2 += conf[i,k-1]

                    area = pesudo_label_batch[i]==0

                    conf_bg = conf_bg/2
                    sam_conf[i,0][area] = conf_bg[area]*outputs_soft[i,0][area].mean()

                    conf_bg_1 = conf_bg_1/2
                    sam_conf_1[i,0][area] = conf_bg_1[area]*outputs_soft_1[i,0][area].mean()

                    conf_bg_2 = conf_bg_2/2
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

                loss_sca = F.cross_entropy(outputs, pgt_sam,ignore_index=4) + F.cross_entropy(outputs, pgt_sam_1,ignore_index=4) + F.cross_entropy(outputs, pgt_sam_2,ignore_index=4)

                if not torch.isnan(loss_sca):
                    loss_SCA = loss_sca
                else:
                    print("loss_SCA is NaN!")
                    loss_SCA = torch.zeros_like(loss_ce)


            else:
                loss_SCA = dice_loss(outputs_soft, pesudo_label_batch.unsqueeze(1)) + dice_loss(outputs_soft_2, pesudo_label_batch.unsqueeze(1)) +dice_loss(outputs_soft_2, pesudo_label_batch.unsqueeze(1)) 



            loss = loss_ce + loss_SCA 


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
                'iteration %d : loss : %f, loss_ce: %f, loss_SCA: %f,loss_ssa:%f alpha: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_SCA.item(),loss_ssa, alpha))

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
                total_dice_scores = [0, 0]
                total_HD95_scores = [0, 0]
                total_IOU_scores = [0, 0]
                num_batches = 0
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
                    with torch.no_grad():
                        outputs = model(volume_batch)[0]
                        preds = torch.argmax(outputs, dim=1)
                        
                        dices, HD95 = calculate_metrics(label_batch, preds, num_classes)
                        for cls in range(num_classes-1):
                            total_dice_scores[cls] += dices[cls]
                            total_HD95_scores[cls] += HD95[cls]

                    num_batches += 1

                avg_dice_scores = [total / num_batches for total in total_dice_scores]
                avg_HD95_scores = [total / num_batches for total in total_HD95_scores]

                avg_dice = np.mean(avg_dice_scores)
                avg_HD95 = np.mean(avg_HD95_scores)

                if avg_dice > best_performance:
                    best_performance,best_hd95 = avg_dice,avg_HD95
                    best_dice_0,best_dice_1 = avg_dice_scores[0],avg_dice_scores[1]
                    best_hd95_0,best_hd95_1 = avg_HD95_scores[0],avg_HD95_scores[1]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    

                logging.info(f'iteration {iter_num} : Mean: Val Dice: {avg_dice:.4f}, Val HD95: {avg_HD95:.4f}')
                logging.info(f'iteration {iter_num} : category 1: Val Dice: {avg_dice_scores[0]:.4f}, Val HD95: {avg_HD95_scores[0]:.4f}')
                logging.info(f'iteration {iter_num} : category 2: Val Dice: {avg_dice_scores[1]:.4f}, Val HD95: {avg_HD95_scores[1]:.4f}')
                
                    
                
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
        writer.writerow(['','mean_dice','mean_hd95', 'Nu_dice','Nu_hd95', 'Cy_dice','Cy_hd95'])
        writer.writerow([args.fold, best_performance,best_hd95,best_dice_0,best_hd95_0,best_dice_1,best_hd95_1])   
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
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path,savepath)
