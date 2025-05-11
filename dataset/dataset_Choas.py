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
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

##微调SAM的Dataset
class BaseDataSets_SAM(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'SAM_PL', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/Chaos_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/Chaos_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        fold1_testing_set = ["patient{:0>2}".format(i) for i in
                        [5,36,13,21]]
        fold1_training_set = ["patient{:0>2}".format(i) for i in  
                              [3, 33, 8, 37, 15, 2, 32, 1, 39, 38, 20, 34, 19, 31, 22, 10]]
        fold2_testing_set = ["patient{:0>2}".format(i) for i in
                        [3, 33, 8, 37]]
        fold2_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 15, 2, 32, 1, 39, 38, 20, 34, 19, 31, 22, 10]]
        
        fold3_testing_set = ["patient{:0>2}".format(i) for i in
                        [15, 2, 32, 1]]
        fold3_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 3, 33, 8, 37, 39, 38, 20, 34, 19, 31, 22, 10]]
        
        fold4_testing_set = ["patient{:0>2}".format(i) for i in
                        [39, 38, 20, 34]]
        fold4_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 3, 33, 8, 37, 15, 2, 32, 1, 19, 31, 22, 10]]
        
        fold5_testing_set = ["patient{:0>2}".format(i) for i in
                        [19, 31, 22, 10]]
        fold5_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 3, 33, 8, 37, 15, 2, 32, 1, 39, 38, 20, 34]]
        
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/Chaos_training_SAM_PL_iteration1/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/Chaos_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        scribble = h5f['scribble'][:]
        if self.split == "train":
            pesudo_label = h5f[self.pesudo_label][:]
            sample = {'image': image, 'label': label, "scribble":scribble,'pesudo_label': pesudo_label}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label,'scribble': scribble}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
def random_rot_flip(image, label,scribble,pesudo_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    pesudo_label = np.rot90(pesudo_label, k)
    scribble = np.rot90(scribble, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    pesudo_label = np.flip(pesudo_label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    return image, label,scribble, pesudo_label
def random_rotate(image, label,scribble, pesudo_label,  cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, 
                           reshape=False)
    scribble = ndimage.rotate(scribble, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    label = ndimage.rotate(label, angle, order=0, 
                           reshape=False)
    pesudo_label = ndimage.rotate(pesudo_label, angle, order=0, 
                           reshape=False)

    return image, label,scribble, pesudo_label
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
            channels, x, y = image.shape
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            scribble = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.uint8))
            scribble = torch.from_numpy(scribble.astype(np.uint8))
            sample = {'image': image, 'label': label,"scribble":scribble}
        return sample


##训练分割网路的Dataset
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'vit_H', pesudo_label_path = "ACDC_training_slices"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        self.pesudo_label_path = pesudo_label_path
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ChaosT1_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ChaosT1_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        fold1_testing_set = ["patient{:0>2}".format(i) for i in
                        [5,36,13,21]]
        fold1_training_set = ["patient{:0>2}".format(i) for i in  
                              [3, 33, 8, 37, 15, 2, 32, 1, 39, 38, 20, 34, 19, 31, 22, 10]]
        # fold1_training_set = ["patient{:0>2}".format(i) for i in  
        #                       [3, 33, 8, 37, 15, 2, 32, 1, 39, 38, 20, 34, 19, 31, 22, 10,5,36,13,21]]
        


        fold2_testing_set = ["patient{:0>2}".format(i) for i in
                        [3, 33, 8, 37]]
        fold2_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 15, 2, 32, 1, 39, 38, 20, 34, 19, 31, 22, 10]]
        
        fold3_testing_set = ["patient{:0>2}".format(i) for i in
                        [15, 2, 32, 1]]
        fold3_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 3, 33, 8, 37, 39, 38, 20, 34, 19, 31, 22, 10]]
        
        fold4_testing_set = ["patient{:0>2}".format(i) for i in
                        [39, 38, 20, 34]]
        fold4_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 3, 33, 8, 37, 15, 2, 32, 1, 19, 31, 22, 10]]
        
        fold5_testing_set = ["patient{:0>2}".format(i) for i in
                        [19, 31, 22, 10]]
        fold5_training_set = ["patient{:0>2}".format(i) for i in  
                              [5, 36, 13, 21, 3, 33, 8, 37, 15, 2, 32, 1, 39, 38, 20, 34]]
        
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ChaosT1_training_conf_iteration0/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ChaosT1_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pesudo_label = h5f[self.pesudo_label][:]
            SAM_conf = np.array([h5f['LIV'][:],h5f['RK'][:],h5f['LK'][:],h5f['SPL'][:]])
            sample = {'image': image, 'label': label, 'scribble': scribble, 'pesudo_label': pesudo_label,'conf':SAM_conf}
            sample = self.transform(sample)

        else:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
def random_rot_flip_conf(image, label, scribble,pesudo_label,conf):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    scribble = np.rot90(scribble, k)
    pesudo_label = np.rot90(pesudo_label, k)
    conf0 = np.rot90(conf[0], k)
    conf1 = np.rot90(conf[1], k)
    conf2 = np.rot90(conf[2], k)
    conf3 = np.rot90(conf[3], k)
    


    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    pesudo_label = np.flip(pesudo_label, axis=axis).copy()
    conf0 = np.flip(conf0, axis=axis).copy()
    conf1 = np.flip(conf1, axis=axis).copy()
    conf2 = np.flip(conf2, axis=axis).copy()
    conf3 = np.flip(conf3, axis=axis).copy()
    conf_new = np.array([conf0,conf1,conf2,conf3])
    
    return image, label, scribble, pesudo_label,conf_new
def random_rotate_conf(image, label, scribble, pesudo_label,conf,  cval):
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
    conf[3] = ndimage.rotate(conf[3], angle, order=0, 
                           reshape=False)

    return image, label, scribble, pesudo_label,conf
class RandomGenerator(object):
    def __init__(self, output_size,split):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        
        if self.split == 'train':
            image, label, scribble, pesudo_label,conf  = sample["image"], sample["label"], sample["scribble"], sample["pesudo_label"],sample["conf"]
            # ind = random.randrange(0, img.shape[0])

            # image = img[ind, ...]
            # label = lab[ind, ...]
            flag = 0
            if random.random() > 0.5:
                image, label, scribble, pesudo_label,conf = random_rot_flip_conf(image, label, scribble, pesudo_label,conf)
                flag=1
            elif random.random() > 0.5:
                if 4 in np.unique(scribble):
                    image, label, scribble, pesudo_label,conf = random_rotate_conf(image, label, scribble, pesudo_label,conf, cval=4)
                else:
                    image, label, scribble, pesudo_label,conf = random_rotate_conf(image, label, scribble, pesudo_label,conf, cval=0)
            x, y= image.shape 
            _,x_conf,y_conf = conf.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            pesudo_label = zoom(pesudo_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            conf = zoom(conf, (1,self.output_size[0] / x_conf, self.output_size[1] / y_conf), order=0)
            

            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            # image_sam = torch.from_numpy(image_sam.astype(np.float32)).permute(2, 0, 1)
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


##分割网路为SAM生成伪标签的Dataset
class BaseDataSets_Net_G_SAM_PL(Dataset):
    def __init__(self, base_dir=None, transform=None):
        self.sample_list = []
        self.transform = transform
        self._base_dir = base_dir
        train_ids, test_ids = self._get_fold_ids()
    
        self.all_slices = os.listdir(
            self._base_dir + "/ACDC_training_slices")
        self.sample_list = []
        for ids in train_ids:
            new_data_list = list(filter(lambda x: re.match(
                '{}.*'.format(ids), x) != None, self.all_slices))
            self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = []
        training_set = [
            i for i in all_cases_set if i not in testing_set]

        return [training_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        h5f = h5py.File(self._base_dir +"/ACDC_training_slices/{}".format(case), 'r')
            
        image = h5f['image'][:]  
        label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        sample["idx"] = case
        return sample    
class RandomGenerator_Net_G_SAM_PL(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label  = sample["image"], sample["label"]

        x, y  = image.shape

        zoom_factors = (self.output_size[0] / x, self.output_size[1] / y)  # 不缩放通道维度
        image = zoom(image, zoom_factors, order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 假设标签是单通道的
    
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {"image": image, "label": label}
        return sample
##SAM为分割网路生成伪标签的Dataset
class BaseDataSets_SAM_pred(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'vit_H', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold1_testing_set = [
        #     "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_testing_set = []
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        image = np.array([image, image, image]).transpose(1, 2, 0) 
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]    
            sample = {'image': image, 'label': label, 'scribble': scribble}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
    

class RandomGenerator_SAM_pred(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label, scribble  = sample["image"], sample["label"], sample["scribble"]
        
        x, y, _= image.shape
        # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y , 1), order=0)
        # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        
        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))

        sample = {"image": image, "label": label, "scribble":scribble}
        return sample

if __name__ == "__main__":
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    train_dataset = BaseDataSets_SAM_pred(base_dir="/home/cj/code/SAM_Scribble/data/ACDC", split="train", transform=transforms.Compose([
        BaseDataSets_SAM_pred([256, 256],'train')]
        ), fold='fold1', sup_type='scribble', edge_paras="30_40_0",pesudo_label='vit_H')

    print(train_dataset[13]['image'].shape)
    # print(np.unique(train_dataset[1]['image'].cpu().detach().numpy()))

    print(train_dataset[13]['label'].shape)
    print(np.unique(train_dataset[1]['label']))

    print(train_dataset[13]['scribble'].shape)
    print(np.unique(train_dataset[1]['scribble']))

    print(train_dataset[13]['pesudo_label'].shape)
    print(np.unique(train_dataset[1]['pesudo_label']))
