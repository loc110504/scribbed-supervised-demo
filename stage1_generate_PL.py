#!/usr/bin/env python3
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor
from dataset.dataset_ACDC import BaseDataSets_SAM_pred, RandomGenerator_SAM_pred
from my_utils.Sampling_Combine import contour_sample, combine
from torchvision.transforms import transforms

def save_pseudo_label(base_dir, iteration, case_id,
                      image_arr, label_arr, scribble_arr, pseudo_mask):
    """
    Lưu file .h5 với các dataset:
      - image: ảnh gốc (HxW)
      - label: ground-truth (HxW)
      - scribble: scribble mask (HxW)
      - SAM_PL: pseudo-label (HxW)
    """
    pseudo_dir = os.path.join(base_dir, f"ACDC_training_SAM_PL_iteration{iteration}")
    os.makedirs(pseudo_dir, exist_ok=True)
    out_path = os.path.join(pseudo_dir, case_id)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('image',    data=image_arr,    compression='gzip')
        f.create_dataset('label',    data=label_arr,    compression='gzip')
        f.create_dataset('scribble', data=scribble_arr, compression='gzip')
        f.create_dataset('SAM_PL',   data=pseudo_mask.astype(np.uint8),
                                          compression='gzip')

def main():
    # ==== Cấu hình ====
    base_dir    = "/scribbed-supervised-demo/datasets/ACDC"
    checkpoint  = "/scribbed-supervised-demo/SAM_Finetune/sam_vit_h_4b8939.pth"
    model_type  = "vit_h"
    iteration   = 1    # đánh số vòng pseudo-labeling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam   = sam_model_registry[model_type](checkpoint).to(device)
    predictor = SamPredictor(sam)

    # ==== Dataset Stage 1 ====
    dataset = BaseDataSets_SAM_pred(
        base_dir=base_dir,
        split="train",
        fold='fold1',
        sup_type='scribble',
        pseudo_label='SAM_PL',
        transform=transforms.Compose([
            RandomGenerator_SAM_pred([256, 256])
        ])
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    num_points = 10  # số điểm prompt mỗi class

    for batch in dataloader:
        # -- Đọc dữ liệu từ batch --
        image_t   = batch['image'].to(device)   # [1, 3, H, W]
        scribble_t= batch['scribble'].to(device) # [1, H, W]
        label_t   = batch['label'].to(device)   # [1, H, W]
        case_id   = batch['idx'][0]             # ví dụ "patient001_slice005.h5"

        # Chuẩn bị numpy arrays để lưu
        orig_image = image_t[0,0].cpu().numpy()   # [H, W] (1 channel)
        label_arr  = label_t[0].cpu().numpy()     # [H, W]
        scribble_arr = scribble_t[0].cpu().numpy()# [H, W]

        # Tạo input 3-channel cho SAM
        image_np = (orig_image * 255).astype(np.uint8)
        image_np = np.stack([image_np]*3, axis=2) # [H, W, 3]

        # -- Sinh điểm prompt bằng contour sampling (truyền Tensor!) --
        pts_lv   = contour_sample(scribble_t, 1, num_points)
        pts_myo  = contour_sample(scribble_t, 2, num_points)
        pts_rv   = contour_sample(scribble_t, 3, num_points)
        pts_bg   = contour_sample(scribble_t, 4, num_points)

        all_pts_lv,  all_lbls_lv, \
        all_pts_rv,  all_lbls_rv, \
        all_pts_myo, all_lbls_myo = combine(pts_lv, pts_rv, pts_myo, pts_bg)

        # -- Sinh mask mỗi class với SAM Predictor --
        predictor.set_image(image_np)
        masks_lv,  _, _ = predictor.predict(
            point_coords = np.array(all_pts_lv)  if all_pts_lv  is not None else None,
            point_labels = np.array(all_lbls_lv) if all_lbls_lv is not None else None,
            multimask_output=False
        )
        predictor.set_image(image_np)
        masks_rv,  _, _ = predictor.predict(
            point_coords = np.array(all_pts_rv)  if all_pts_rv  is not None else None,
            point_labels = np.array(all_lbls_rv) if all_lbls_rv is not None else None,
            multimask_output=False
        )
        predictor.set_image(image_np)
        masks_myo, _, _ = predictor.predict(
            point_coords = np.array(all_pts_myo)  if all_pts_myo  is not None else None,
            point_labels = np.array(all_lbls_myo) if all_lbls_myo is not None else None,
            multimask_output=False
        )

        # -- Kết hợp thành pseudo-mask đa lớp --
        h, w = masks_lv.shape[1], masks_lv.shape[2]
        pseudo_mask = np.zeros((h, w), dtype=np.uint8)
        if all_pts_myo is not None:
            pseudo_mask[masks_myo[0]] = 2
        if all_pts_lv is not None:
            pseudo_mask[masks_lv[0]]  = 1
        if all_pts_rv is not None:
            pseudo_mask[masks_rv[0]]  = 3

        # -- Lưu file .h5 cho Stage 2 & 3 --
        save_pseudo_label(
            base_dir, iteration, case_id,
            image_arr    = orig_image,
            label_arr    = label_arr,
            scribble_arr = scribble_arr,
            pseudo_mask  = pseudo_mask
        )

if __name__ == "__main__":
    main()
