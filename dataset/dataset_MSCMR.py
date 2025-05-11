import itertools
import os
import random
import re
from glob import glob

# import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from random import sample

import pandas as pd

class MSCMRDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",
                 train_dir="/MSCMR_training_conf_iteration1", val_dir="/MSCMR_training_volumes", pl = 'vit_H'):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pl = pl
        train_ids, test_ids = self._get_fold_ids(fold)
        

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            # print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        training_set = ["subject{:0>2}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        # validation_set = ["subject{:0>2}".format(i) for i in [1, 8,29,36,41]]
        validation_set = ["subject{:0>2}".format(i) for i in [3,5,10,11,12,16,17,23,28,30,33,35,38,40,43]]
        return [training_set, validation_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pesudo_label = h5f[self.pl][:]
            SAM_conf = np.array([h5f['LV'][:],h5f['MYO'][:],h5f['RV'][:]])
            sample = {'image': image, 'label': label, 'scribble': scribble, 'pesudo_label': pesudo_label,'conf':SAM_conf}
            sample = self.transform(sample)

        else:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
    
    


def random_rot_flip(image, label, scribble,pesudo_label,conf):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    scribble = np.rot90(scribble, k)
    pesudo_label = np.rot90(pesudo_label, k)
    conf0 = np.rot90(conf[0], k)
    conf1 = np.rot90(conf[1], k)
    conf2 = np.rot90(conf[2], k)
    


    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    pesudo_label = np.flip(pesudo_label, axis=axis).copy()
    conf0 = np.flip(conf0, axis=axis).copy()
    conf1 = np.flip(conf1, axis=axis).copy()
    conf2 = np.flip(conf2, axis=axis).copy()
    conf_new = np.array([conf0,conf1,conf2])
    
    return image, label, scribble, pesudo_label,conf_new

def random_rotate(image, label, scribble, pesudo_label,conf,  cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, 
                           reshape=False)
    label = ndimage.rotate(label, angle, order=0, 
                           reshape=False)
    scribble = ndimage.rotate(scribble, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    pesudo_label = ndimage.rotate(pesudo_label, angle, order=0, 
                           reshape=False)
    conf[0] = ndimage.rotate(conf[0], angle, order=0, 
                           reshape=False)
    conf[1] = ndimage.rotate(conf[1], angle, order=0, 
                           reshape=False)
    conf[2] = ndimage.rotate(conf[2], angle, order=0, 
                           reshape=False)

    return image, label, scribble, pesudo_label,conf



class RandomGenerator(object):
    # def __init__(self, output_size):
    #     self.output_size = output_size

    # def __call__(self, sample):
    #     image, label, scribble = sample["image"], sample["label"], sample["scribble"]
    #     # ind = random.randrange(0, img.shape[0])
    #     # image = img[ind, ...]
    #     # label = lab[ind, ...]
    #     if random.random() > 0.5:
    #         image, label, scribble = random_rot_flip(image, label, scribble)
    #     elif random.random() > 0.5:
    #         if 4 in np.unique(label):
    #             image, label, scribble = random_rotate(image, label, scribble, cval=4)
    #         else:
    #             image, label, scribble = random_rotate(image, label, scribble, cval=0)
    #     x, y = image.shape
    #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    #     scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    #     image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    #     label = torch.from_numpy(label.astype(np.uint8))
    #     scribble = torch.from_numpy(scribble.astype(np.uint8))
    #     sample = {"image": image, "label": label, "scribble":scribble}
    #     return sample
    def __init__(self, output_size,split):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        
        if self.split == 'train':
            image, label, scribble, pesudo_label, conf  = sample["image"], sample["label"], sample["scribble"], sample["pesudo_label"],sample["conf"]


            if random.random() > 0.5:
                image, label, scribble, pesudo_label,conf = random_rot_flip(image, label, scribble, pesudo_label,conf)
            elif random.random() > 0.5:
                if 4 in np.unique(scribble):
                    image, label, scribble, pesudo_label,conf = random_rotate(image, label, scribble, pesudo_label,conf, cval=4)
                else:
                    image, label, scribble, pesudo_label,conf = random_rotate(image, label, scribble, pesudo_label,conf, cval=0)
            x, y= image.shape
            _,x_conf, y_conf= conf.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            pesudo_label = zoom(pesudo_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            conf = zoom(conf, (1,self.output_size[0] / x_conf, self.output_size[1] / y_conf), order=0)

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            pesudo_label = torch.from_numpy(pesudo_label.astype(np.uint8))
            conf = torch.from_numpy(conf.astype(np.float32))
            scribble = torch.from_numpy(scribble.astype(np.uint8))

            sample = {"image": image, "label": label, "scribble":scribble, 'pesudo_label': pesudo_label,'conf':conf}
        else:
            image, label = sample['image'], sample['label']
            channels, x, y = image.shape
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.uint8))
            sample = {'image': image, 'label': label}
        return sample


class MSCMR_BaseDataSets_SAM_pred(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",train_dir="/MSCMR_training_slices", val_dir="/MSCMR_training_volumes"):
        
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        train_ids, test_ids = self._get_fold_ids(fold)
    

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        training_set = ["subject{:0>2}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        validation_set = ["subject{:0>2}".format(i) for i in [1, 29, 36, 41, 8]]
        return [training_set, validation_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        image = h5f['image'][:]  
        label = h5f['label'][:]
        scribble = h5f['scribble'][:]
        if self.split == "train":
            image_3 = np.array([image,image,image]).transpose(1,2,0)
            sample = {'image': image_3, 'image_sam': image_3, 'label': label, 'scribble': scribble}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label, "scribble": scribble}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
    

class MSCMR_RandomGenerator_SAM_pred(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label, scribble  = sample["image"], sample["label"], sample["scribble"]


        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))

        sample = {"image": image, "label": label, "scribble":scribble}
        return sample

class BaseDataSets_SAM(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",
                 train_dir="/MSCMR_training_SAM_PL_iteration2", val_dir="/MSCMR_training_volumes",pesudo_label = 'SAM_PL'):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)

        

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))
    def _get_fold_ids(self, fold):
        training_set = ["subject{:0>2}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        validation_set = ["subject{:0>2}".format(i) for i in [1, 3,5,8,10,11,12,16,17,23,28,29,30,33,35,36,38,40,41,43]]
        # validation_set = ["subject{:0>2}".format(i) for i in [1, 8,29,36,41]]
        # validation_set = ["subject{:0>2}".format(i) for i in [3,5,10,11,12,16,17,23,28,30,33,35,38,40,43]]
        return [training_set, validation_set]


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pesudo_label = h5f[self.pesudo_label][:]
            sample = {'image': image, 'label': label, "scribble":scribble,'pesudo_label': pesudo_label}
            sample = self.transform(sample)
        else:
            scribble = h5f['label'][:]
            sample = {'image': image, 'label': label,'scribble': scribble}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
    


class RandomGenerator_SAM(object):
    def __init__(self, output_size,split):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        
        if self.split == 'train':
            image, label, scribble,pesudo_label  = sample["image"], sample["label"],sample["scribble"],sample["pesudo_label"]
            # if random.random() > 0.5:
            #     image, label, scribble, pesudo_label = random_rot_flip(image, label, scribble, pesudo_label)
            # elif random.random() > 0.5:
            #     if 4 in np.unique(scribble):
            #         image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=4)
            #     else:
            #         image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=0)
            x, y= image.shape
            x_pl,y_pl = pesudo_label.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            pesudo_label = zoom(pesudo_label, (self.output_size[0] / x_pl, self.output_size[1] / y_pl), order=0)

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            pesudo_label = torch.from_numpy(pesudo_label.astype(np.uint8))
            scribble = torch.from_numpy(scribble.astype(np.uint8))
            sample = {"image": image, "label": label,"scribble":scribble, 'pesudo_label': pesudo_label}
        else:
            image, label,scribble = sample['image'], sample['label'], sample["scribble"]
            # channels, x, y = image.shape
            # image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            # label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            # scribble = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.uint8))
            scribble = torch.from_numpy(scribble.astype(np.uint8))
            sample = {'image': image, 'label': label,"scribble":scribble}
        return sample



if __name__ == "__main__":
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    train_dataset = BaseDataSets_SAM_pred(base_dir="/home/cj/code/SAM_Scribble/data/ACDC", split="train", transform=transforms.Compose([
        BaseDataSets_SAM_pred([256, 256],'train')]
        ), fold='fold1', sup_type='scribble', edge_paras="30_40_0",pesudo_label='vit_H')
    
    print(train_dataset[1]['image'].shape)
    print(np.unique(train_dataset[1]['image'].cpu().detach().numpy()))
    
    print(train_dataset[1]['label'].shape)
    print(np.unique(train_dataset[1]['label'].cpu().detach().numpy()))

    print(train_dataset[1]['scribble'].shape)
    print(np.unique(train_dataset[1]['scribble'].cpu().detach().numpy()))