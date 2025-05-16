#!/usr/bin/env python3
import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch_scatter

from losses.pd_loss import pDLoss
from dataset.dataset_ACDC import BaseDataSets, RandomGenerator
from Networks.net_factory import net_factory
from my_utils.val2D import test_single_volume_cct

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',    type=str,   default='datasets/ACDC',             help='Thư mục gốc dữ liệu')
    parser.add_argument('--exp',          type=str,   default='ACDC/InterFA_SA_Two',       help='Tên experiment')
    parser.add_argument('--fold',         type=str,   default='fold1',                     help='Fold CV')
    parser.add_argument('--sup_type',     type=str,   default='scribble',                  help='Loại supervision')
    parser.add_argument('--model',        type=str,   default='unet_DMPLS_att',            help='Tên model')
    parser.add_argument('--num_classes',  type=int,   default=4,                           help='Số class')
    parser.add_argument('--max_iterations', type=int, default=30000,                      help='Số bước train tối đa')
    parser.add_argument('--batch_size',   type=int,   default=16,                          help='Batch size')
    parser.add_argument('--deterministic',type=int,   default=1,                           help='Reproducible hay không')
    parser.add_argument('--base_lr',      type=float, default=0.01,                        help='Learning rate')
    parser.add_argument('--patch_size',   type=list,  default=[256,256],                   help='Kích thước patch')
    parser.add_argument('--seed',         type=int,   default=2022,                        help='Random seed')
    parser.add_argument('--T',            type=int,   default=1,                           help='SSA temperature')
    parser.add_argument('--PL_path',      type=str,   default='ACDC_training_SAM_PL_iteration1', help='Thư mục pseudo-label')
    parser.add_argument('--ca_iteration', type=int,   default=10000,                      help='Iteration bắt đầu dùng SCA loss')
    parser.add_argument('--pseudo_label', type=str, default='SAM_PL', help='Key của pseudo-label trong .h5')
    return parser.parse_args()

def tv_loss(pred):
    min_pool = F.max_pool2d(pred * -1, (3,3), 1,1)*-1
    contour  = F.relu(F.max_pool2d(min_pool, (3,3),1,1) - min_pool)
    return torch.mean(torch.abs(contour))

def train(args, snapshot_path, savepath, device):
    # --- chuẩn bị model và data ---
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    model = model.to(device)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train",
        transform=transforms.Compose([RandomGenerator(args.patch_size, split='train')]),
        fold=args.fold, sup_type=args.sup_type, pseudo_label=args.pseudo_label, pseudo_label_path=args.PL_path
    )
    db_val = BaseDataSets(
        base_dir=args.root_path, split="val",
        transform=transforms.Compose([RandomGenerator(args.patch_size, split='val')]),
        fold=args.fold
    )

    trainloader = DataLoader(
        db_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=lambda wid: random.seed(args.seed+wid), drop_last=True
    )
    valloader = DataLoader(
        db_val, batch_size=1, shuffle=False,
        num_workers=2, drop_last=False
    )

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=1e-4)
    ce_loss   = CrossEntropyLoss(ignore_index=4)
    dice_loss = pDLoss(args.num_classes, ignore_index=4)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_perf = 0.0

    model.train()
    for epoch in tqdm(range(max_epoch), desc="Epochs"):
        for batch in trainloader:
            # --- load batch sang device ---
            imgs   = batch['image'].to(device)       # [B,1,H,W]
            labels = batch['label'].to(device)       # [B,H,W]
            scrib  = batch['scribble'].to(device)    # [B,H,W]
            pls    = batch['SAM_PL'].to(device)      # [B,H,W]
            conf   = batch['conf'].to(device)        # [B,3,H,W]

            # --- forward ---
            outputs, mainseg, auxseg, latent = model(imgs)
            out_soft  = F.softmax(outputs, dim=1)
            out_soft1 = F.softmax(mainseg, dim=1)
            out_soft2 = F.softmax(auxseg, dim=1)

            # --- CE loss ---
            lc1 = ce_loss(outputs, labels.long())
            lc2 = ce_loss(mainseg, labels.long())
            lc3 = ce_loss(auxseg, labels.long())
            loss_ce = lc1 + lc2 + lc3

            # --- SSA loss ---
            B, C, Hf, Wf = latent.shape
            feat = F.normalize(latent, dim=1).view(B, C, -1)
            idx  = pls.view(B,1,-1).long()
            proto = torch_scatter.scatter_mean(feat.detach(), idx, dim=2)
            proto = F.normalize(proto, dim=1)
            idx_s = idx.squeeze(1)
            logits = torch.bmm(proto.permute(0,2,1), feat) * args.T
            loss_ssa = F.cross_entropy(logits, idx_s, ignore_index=0)

            # --- SCA loss ---
            if iter_num >= args.ca_iteration:
                # build pgt_sam, pgt_sam_1, pgt_sam_2 từ conf & out_soft...
                sam_conf = -1e5*torch.ones_like(outputs)
                sam_conf1 = sam_conf.clone()
                sam_conf2 = sam_conf.clone()
                bg_accum = torch.zeros((Hf,Wf), device=device)
                for b in range(B):
                    for k in range(1, args.num_classes):
                        area = (pls[b]==k)
                        sam_conf[b,k][area]  = conf[b,k-1][area] * out_soft[b,k][area].mean()
                        sam_conf1[b,k][area] = conf[b,k-1][area] * out_soft1[b,k][area].mean()
                        sam_conf2[b,k][area] = conf[b,k-1][area] * out_soft2[b,k][area].mean()
                        bg_accum += conf[b,k-1]
                    area0 = (pls[b]==0)
                    bg_accum /= (args.num_classes-1)
                    sam_conf[b,0][area0]  = bg_accum[area0]  * out_soft[b,0][area0].mean()
                    sam_conf1[b,0][area0] = bg_accum[area0]  * out_soft1[b,0][area0].mean()
                    sam_conf2[b,0][area0] = bg_accum[area0]  * out_soft2[b,0][area0].mean()

                pgt_sam   = sam_conf.argmax(dim=1)
                pgt_sam1  = sam_conf1.argmax(dim=1)
                pgt_sam2  = sam_conf2.argmax(dim=1)
                loss_sca = (
                    ce_loss(outputs, pgt_sam) +
                    ce_loss(outputs, pgt_sam1) +
                    ce_loss(outputs, pgt_sam2)
                )
            else:
                loss_sca = (
                    dice_loss(out_soft.unsqueeze(1),  pls.unsqueeze(1)) +
                    dice_loss(out_soft1.unsqueeze(1), pls.unsqueeze(1)) +
                    dice_loss(out_soft2.unsqueeze(1), pls.unsqueeze(1))
                )

            loss = loss_ce + loss_ssa + loss_sca

            # --- backward + update ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            # --- log to tensorboard ---
            writer.add_scalar('train/loss_total', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce',    loss_ce.item(), iter_num)

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

        # --- validation per-epoch ---
        model.eval()
        metrics = []
        with torch.no_grad():
            for val_batch in valloader:
                metrics.append(
                    test_single_volume_cct(
                        val_batch['image'].to(device),
                        val_batch['label'].to(device),
                        model,
                        classes=args.num_classes
                    )
                )
        mean_metrics = np.mean(np.stack(metrics), axis=0)
        mean_dice  = mean_metrics[:,0].mean()
        writer.add_scalar('val/mean_dice', mean_dice, iter_num)
        if mean_dice > best_perf:
            best_perf = mean_dice
            torch.save(model.state_dict(),
                       os.path.join(snapshot_path, 'best_model.pth'))
        model.train()

    writer.close()
    # lưu bảng metrics cuối cùng
    with open(os.path.join(savepath, 'best_metrics.csv'), 'a') as f:
        f.write(f"{args.fold},{best_perf:.4f}\n")
    return

if __name__ == "__main__":
    args = parse_args()
    # deterministic
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savepath     = os.path.join("model_New", args.exp)
    snapshot_path= os.path.join(savepath, args.fold, args.sup_type)
    os.makedirs(snapshot_path, exist_ok=True)
    # chạy train
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s'
    )
    train(args, snapshot_path, savepath, device)
