""""
Rewriting the data loader , loss and other functions here.
Rewrote the  whole script due to directory name error lol!
"""

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    Invertd,
    SaveImaged
)

import h5py
import numpy as np
import os
import nibabel as nib
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
import cc3d
import fastremap
from scipy import ndimage
from monai.data import decollate_batch

DEFAULT_POST_FIX = PostFix.meta()

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Aorta', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 'Colon Tumor', 'Kidney Cyst']

TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '01_2': [1,3,4,5,6,7,11,14],
    '02': [1,3,4,5,6,7,11,14],
    '03': [6],
    '04': [6,27], # post process
    '05': [2,3,26,32], # post process
    '06': [1,2,3,4,6,7,11,16,17],
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,13,20,21,23,24],
    '08': [6, 2, 3, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,17,2,3],  
    '13': [6,2,3,1,11,8,9,7,4,5,12,13,25], 
#    '14': [11, 28],
    '10_03': [6, 27], # post process
    '10_06': [30],
    '10_07': [11, 28], # post process
    '10_08': [15, 29], # post process
    '10_09': [1],
    '10_10': [31],
    '14':[1,2,3,4,6,7,11],
#    '15': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] ## total segmentation
    '21':[1,2,3,6,11]
}

# TEMPLATE = {
#     '10_03': [2, 5], # post process
#     '10_06': [8],
#     '10_07': [3, 6], # post process
#     '10_08': [4, 7], # post process
#     '10_09': [1],
#     '10_10': [9],
# }

TUMOR_ORGAN = {
    'Kidney Tumor': [2,3], 
    'Liver Tumor': [6], 
    'Pancreas Tumor': [11], 
    'Hepatic Vessel Tumor': [15], 
    'Lung Tumor': [16,17], 
    'Colon Tumor': [18], 
    'Kidney Cyst': [2,3]
}

MERGE_MAPPING_v1 = {
    '01': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14)],
    '01_2': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14)],
    '02': [(1,1), (3,3), (4,4), (5,5), (6,6), (7,7), (11,11), (14,14)],
    '03': [(6,1)],
    '04': [(6,1), (27,2)],
    '05': [(2,1), (3,1), (26, 2), (32,3)],
    '06': [(1,1), (2,2), (3,3), (4,4), (6,5), (7,6), (11,7), (16,8), (17,9)],
    '07': [(1,2), (2,4), (3,3), (4,6), (5,7), (6,1), (7,5), (11,8), (12,12), (13,12), (14,9), (18,10), (19,11), (20,13), (21,14), (23,15), (24,16)],
    '08': [(1,3), (2,2), (3,2), (6,1), (11,4)],
    '09': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (11,10), (12,11), (13,12), (14,13), (21,14), (22,15)],
    '10_03': [(6,1), (27,2)],
    '10_06': [(30,1)],
    '10_07': [(11,1), (28,2)],
    '10_08': [(15,1), (29,2)],
    '10_09': [(1,1)],
    '10_10': [(31,1)],
    '12': [(2,4), (3,4), (21,2), (6,1), (16,3), (17,3)],  
    '13': [(1,3), (2,2), (3,2), (4,8), (5,9), (6,1), (7,7), (8,5), (9,6), (11,4), (12,10), (13,11), (25,12)],
    '15': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14), (16,16), (17,17), (18,18)],
    '21': [(6,1),(2,2),(3,2),(1,3),(11,4)]
}

NUM_CLASS = 32

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key


def get_key_2(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]

    elif dataset_index == 1:
        if int(name[-2:]) >= 60:
            template_key = '01_2'
        else:
            template_key = '01'
    

    else:
        template_key = name[0:2]
    return template_key


def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)


    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision
    
def dice_score_2(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape == labels.shape, "predict & target batch size don't match"
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den

    return dice

def dice_score_np(preds, labels, spe_sen=False):  # on CPU with NumPy
    ### preds: w,h,d; labels: w,h,d
    assert preds.shape == labels.shape, "predict & target batch size don't match"
    
    # Flattening the arrays
    predict = preds.ravel()
    target = labels.ravel()

    # True positives
    tp = np.sum(predict * target)

    # Denominator: sum of predicted and target pixels + 1 to avoid division by zero
    den = np.sum(predict) + np.sum(target) + 1

    # Dice score calculation
    dice = 2 * tp / den

    return dice

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['post_label'] = data[0]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key not in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

def custom_nested_collate_2_val_test(batch):

    flat_batch = batch
    
    def collate_dict_list(dict_list):
        if not dict_list:
            return {}
        elem = dict_list[0]
        if isinstance(elem, Mapping):
            return {key: collate_dict_list([d[key] for d in dict_list if key in d]) for key in elem}
        elif isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            it = iter(dict_list)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                return dict_list
            return [collate_dict_list(samples) for samples in zip(*dict_list)]
        elif isinstance(elem, (np.ndarray, np.number)):
            return np.stack(dict_list)
        elif isinstance(elem, torch.Tensor):
            return torch.stack(dict_list)
        else:
            try:
                return list_data_collate(dict_list)
            except:
                return dict_list

    # Collate each sample in the batch
    return collate_dict_list(flat_batch)

def custom_nested_collate_2(batch):
    # this cause of num_samples.
    flat_batch = [item for pair in batch for item in pair]
    
    def collate_dict_list(dict_list):
        if not dict_list:
            return {}
        elem = dict_list[0]
        if isinstance(elem, Mapping):
            return {key: collate_dict_list([d[key] for d in dict_list if key in d]) for key in elem}
        elif isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            it = iter(dict_list)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                return dict_list
            return [collate_dict_list(samples) for samples in zip(*dict_list)]
        elif isinstance(elem, (np.ndarray, np.number)):
            return np.stack(dict_list)
        elif isinstance(elem, torch.Tensor):
            return torch.stack(dict_list)
        else:
            try:
                return list_data_collate(dict_list)
            except:
                return dict_list

    # Collate each sample in the batch
    return collate_dict_list(flat_batch)
    
def get_train_val_data_loader(args):

    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []

    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_train.txt'):
            name = line.strip().split()[1].split('.')[0]
            train_img.append(args.data_root_path + line.strip().split()[0])
            train_lbl.append(args.data_root_path + line.strip().split()[1])
            train_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            train_name.append(name)
    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))

    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_val.txt'):
            name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))

    train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)

    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    val_sampler = DistributedSampler(dataset = val_dataset, even_divisible=True,shuffle=False) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                            collate_fn=custom_nested_collate_2, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=(val_sampler is None), num_workers=args.num_workers, collate_fn=custom_nested_collate_2_val_test,sampler = val_sampler)

    return train_loader,val_loader,train_sampler,val_sampler

def get_test_data_loader(args):

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )

    #get the dictionary of the files from the files list.
    test_img = []
    test_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_test2.txt'):
            name = line.strip().split()[0].split('.')[0]
            test_img.append(args.data_root_path + line.strip().split()[0])
            test_name.append(name)
    data_dicts_test = [{'image': image,'name': name}
                for image, name in zip(test_img, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=list_data_collate)
    return test_loader, test_transforms

def get_test_txt_loader(args):

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )

    #get the dictionary of the files from the files list.
    test_img = []
    test_name = []
    test_prompt = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_test2.txt'):
            name = line.strip().split()[0].split('.')[0]
            test_img.append(args.data_root_path + line.strip().split()[0])
            test_name.append(name)
            test_prompt.append(get_key(name))
    data_dicts_test = [{'image': image,'name': name, 'prompt':prompt}
                for image, name, prompt in zip(test_img, test_name, test_prompt)]
    print('test len {}'.format(len(data_dicts_test)))

    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=list_data_collate)
    return test_loader, test_transforms

def get_val_txt_loader(args):

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    val_prompts = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_val.txt'):
            name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)
            val_prompts.append(get_key(name))

    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name,'prompt':prompt}
                for image, label, post_label, name, prompt in zip(val_img, val_lbl, val_post_lbl, val_name, val_prompts)]
    print('val len {}'.format(len(data_dicts_val)))

    val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
    val_sampler = None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=(val_sampler is None), num_workers=args.num_workers, collate_fn=custom_nested_collate_2_val_test,sampler = val_sampler)

    return val_loader,val_transforms

def get_test_loader(args):

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    val_prompts = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_test.txt'):
            name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)
            val_prompts.append(get_key(name))

    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name,'prompt':prompt}
                for image, label, post_label, name, prompt in zip(val_img, val_lbl, val_post_lbl, val_name, val_prompts)]
    print('val len {}'.format(len(data_dicts_val)))

    val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
    val_sampler = None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=(val_sampler is None), num_workers=args.num_workers, collate_fn=custom_nested_collate_2_val_test,sampler = val_sampler)

    return val_loader,val_transforms


def get_train_val_txt_loader(args):

    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []
    train_prompts = []

    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_train.txt'):
            name = line.strip().split()[1].split('.')[0]
            train_img.append(args.data_root_path + line.strip().split()[0])
            train_lbl.append(args.data_root_path + line.strip().split()[1])
            train_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            train_name.append(name)
            train_prompts.append(get_key(name))

    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name, 'prompt':prompt}
                for image, label, post_label, name, prompt in zip(train_img, train_lbl, train_post_lbl, train_name,train_prompts)]
    print('train len {}'.format(len(data_dicts_train)))

    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    val_prompts = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_val.txt'):
            name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)
            val_prompts.append(get_key(name))

    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name,'prompt':prompt}
                for image, label, post_label, name, prompt in zip(val_img, val_lbl, val_post_lbl, val_name, val_prompts)]
    print('val len {}'.format(len(data_dicts_val)))

    train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)

    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    val_sampler = DistributedSampler(dataset = val_dataset, even_divisible=True,shuffle=False) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                            collate_fn=custom_nested_collate_2, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=(val_sampler is None), num_workers=args.num_workers, collate_fn=custom_nested_collate_2_val_test,sampler = val_sampler)

    return train_loader,val_loader,train_sampler,val_sampler

def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label

def merge_and_top_organ(pred_mask,organ_list):

    out_mask = np.zeros(pred_mask.shape[1:],np.uint8)
    for organ in organ_list:
        out_mask = np.logical_or(out_mask,pred_mask[organ-1])
    
    out_mask = extract_topk_largest_candidates(out_mask,len(organ_list))
    return out_mask

def organ_region_filter_out(tumor_mask, organ_mask):
    ## dialtion
    organ_mask = ndimage.binary_closing(organ_mask, structure=np.ones((5,5,5)))
    organ_mask = ndimage.binary_dilation(organ_mask, structure=np.ones((5,5,5)))
    ## filter out
    tumor_mask = organ_mask * tumor_mask

    return tumor_mask

def PSVein_post_process(PSVein_mask, pancreas_mask):
    xy_sum_pancreas = pancreas_mask.sum(axis=0).sum(axis=0)
    z_non_zero = np.nonzero(xy_sum_pancreas)
    z_value = np.min(z_non_zero) ## the down side of pancreas
    new_PSVein = PSVein_mask.copy()
    new_PSVein[:,:,:z_value] = 0
    return new_PSVein

def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    ## npy_mask: w, h, d
    ## organ_num: the maximum number of connected component
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    t_mask = npy_mask.copy()
    keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, 1)

    return out_mask

def organ_post_process(pred_mask,organ_list):

    post_pred_mask = np.zeros(pred_mask.shape)
    
    for organ in organ_list:

        if organ == 11:

            post_pred_mask[10] = extract_topk_largest_candidates(pred_mask[10],1)

            if 10 in organ_list:

                post_pred_mask[9] = PSVein_post_process(pred_mask[9],post_pred_mask[10])
        
        elif organ in [1,2,3,4,5,6,7,8,9,12,13,14,18,19,20,21,22,23,24,25]:

            post_pred_mask[organ-1] = extract_topk_largest_candidates(pred_mask[organ-1],1)
        
        elif organ in [26,27]: #for kindey and liver using no tumor predictions for help

            organ_mask = merge_and_top_organ(pred_mask,TUMOR_ORGAN[ORGAN_NAME[organ-1]])
            post_pred_mask[organ-1] = organ_region_filter_out(pred_mask[organ-1],organ_mask)
        
        else:

            post_pred_mask[organ-1] = pred_mask[organ-1]
    
    return post_pred_mask

def debug_merge_process(data, name, stage="unknown"):
    """
    Debug helper to track data transformations during merge process
    Args:
        data: Input data array (ground truth or predictions)
        name: Name of the case being processed
        stage: String identifier for the processing stage
    """
    print(f"\n=== Debug Info for {stage} ===")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Unique values: {np.unique(data)}")
    print(f"Min value: {data.min()}")
    print(f"Max value: {data.max()}")
    
    # Check if data matches expected format
    if len(data.shape) == 4:
        C, H, W, D = data.shape
        print(f"Number of channels (C): {C}")
        print(f"Expected format: (C,H,W,D) = ({C},{H},{W},{D})")
        
        # Check if values are binary for each channel
        for i in range(C):
            unique_vals = np.unique(data[i])
            print(f"Channel {i} unique values: {unique_vals}")
            if not np.array_equal(unique_vals, np.array([0,1])) and not np.array_equal(unique_vals, np.array([0])):
                print(f"Warning: Channel {i} is not binary!")
    else:
        print(f"Warning: Unexpected data shape! Expected 4D array, got {len(data.shape)}D")
    
def verify_merge_result(original, merged, name, template_key):
    """
    Verify the correctness of merge operation
    Args:
        original: Original multi-channel mask
        merged: Merged single-channel mask
        name: Case name
        template_key: Template key used for merging
    """
    print("\n=== Merge Verification ===")
    print(f"Template Key: {template_key}")
    print(f"Original shape: {original.shape}")
    print(f"Merged shape: {merged.shape}")
    
    # Verify mapping
    transfer_mapping = MERGE_MAPPING_v1[template_key]
    expected_values = set(tgt for _, tgt in transfer_mapping)
    actual_values = set(np.unique(merged))
    
    print(f"Expected values after merge: {sorted(expected_values)}")
    print(f"Actual values after merge: {sorted(actual_values)}")
    
    # Check for missing or unexpected values
    missing = expected_values - actual_values
    unexpected = actual_values - expected_values - {0}  # 0 is background
    
    if missing:
        print(f"Warning: Missing expected values: {missing}")
    if unexpected:
        print(f"Warning: Unexpected values present: {unexpected}")

def modified_merge_label_v1(pred_mask, name):
    """
    Modified merge function with debugging
    """
    debug_merge_process(pred_mask, name, "Pre-merge")
    
    C, H, W, D = pred_mask.shape
    merged_label_v1 = np.zeros((1, H, W, D))
    template_key = get_key_2(name)
    transfer_mapping_v1 = MERGE_MAPPING_v1[template_key]
    
    for item in transfer_mapping_v1:
        src, tgt = item
        # Add debug print for each mapping
        print(f"Processing mapping {src} -> {tgt}")
        print(f"Source channel stats: min={pred_mask[src-1].min()}, max={pred_mask[src-1].max()}")
        merged_label_v1[0][pred_mask[src-1] == 1] = tgt
    
    debug_merge_process(merged_label_v1, name, "Post-merge")
    verify_merge_result(pred_mask, merged_label_v1, name, template_key)
    
    return merged_label_v1

def merge_label_v1(pred_mask,name):

    C,H,W,D = pred_mask.shape
    merged_label_v1 = np.zeros((1,H,W,D))
    template_key = get_key_2(name)
    transfer_mapping_v1 = MERGE_MAPPING_v1[template_key]
    organ_index = []
    for item in transfer_mapping_v1:
        src, tgt = item
        merged_label_v1[0][pred_mask[src-1]==1] = tgt
    
    return merged_label_v1

def save_result(batch,input_transform,save_dir):

    post_transforms = Compose([
        Invertd(
            keys = ['result'],
            transform = input_transform,
            orig_keys = ['image'],
            nearest_interp=True,
            to_tensor=True
        ),
        SaveImaged(
            keys='result',
            meta_keys='image_meta_dict',
            output_dir = save_dir,
            output_ext = '.nii.gz',
            output_postfix = '',
            print_log = False,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False
        )
    ])

    batch = [post_transforms(i) for i in decollate_batch(batch)]

def save_result_2(batch, test_transform, save_dir):
    """
    Save prediction results with original spacing and origin.
    
    Args:
        batch: Dictionary containing 'image', 'result', and 'name' keys
        test_transform: Test transform pipeline used during data loading
        save_dir: Directory to save the results
        
    Returns:
        None
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract original image metadata before any transforms
    orig_img = nib.load(batch['image_meta_dict']['filename_or_obj'][0])
    original_spacing = orig_img.header.get_zooms()[:3]
    original_affine = orig_img.affine.copy()
    
    # Create a transform to revert back to original spacing
    revert_spacing = Compose([
        Spacingd(
            keys=["result"],
            pixdim=original_spacing,
            mode="nearest",
            recompute_affine=True
        )
    ])
    
    try:
        # Process each item in batch
        for item in decollate_batch(batch):
            # Ensure prediction is detached from computation graph
            if torch.is_tensor(item["result"]):
                item["result"] = item["result"].detach()
            
            # Remove batch dimension if present
            if len(item["result"].shape) == 5:
                item["result"] = item["result"].squeeze(0)
            elif len(item["result"].shape) == 4:
                item["result"] = item["result"].squeeze()
                
            # Add channel dimension if needed
            if len(item["result"].shape) == 3:
                item["result"] = item["result"][None]
            
            # Add necessary metadata for spacing transform
            item["result_meta_dict"] = {
                "affine": original_affine,
                "original_affine": original_affine,
                "spatial_shape": item["result"].shape[1:],
            }
            
            # Get the filename
            name = item["name"][0] if isinstance(item["name"], (list, tuple)) else item["name"]
            base_name = os.path.basename(name)
            if '.' in base_name:
                base_name = base_name.split('.')[0]
            
            # Convert to numpy while preserving channel dimension
            result_data = item["result"].cpu().numpy()
            
            # Create NIfTI image with original affine
            result_img = nib.Nifti1Image(result_data[0], original_affine)  # Remove channel dim
            
            # Set header information
            header = result_img.header
            header.set_zooms(original_spacing)
            header.set_data_dtype(np.uint8)
            
            # Save the image
            output_path = os.path.join(save_dir, f"{base_name}.nii.gz")
            nib.save(result_img, output_path)
            
            # Verify the saved file
            saved_img = nib.load(output_path)
            if not np.allclose(saved_img.header.get_zooms()[:3], original_spacing):
                print(f"Warning: Spacing mismatch for {base_name}")
                print(f"Expected: {original_spacing}")
                print(f"Got: {saved_img.header.get_zooms()[:3]}")
            if not np.allclose(saved_img.affine[:3, 3], original_affine[:3, 3]):
                print(f"Warning: Origin mismatch for {base_name}")
                print(f"Expected: {original_affine[:3, 3]}")
                print(f"Got: {saved_img.affine[:3, 3]}")
            
    except Exception as e:
        print(f"Error during saving: {str(e)}")
        raise

#save organs in seperate folder for the given prediction 
def save_result_organwise(batch, save_dir, input_transform, organ_list):
    ### function: save the prediction result into dir
    ## Input
    ## batch: the batch dict output from the monai dataloader
    ## one_channel_label: the predicted reuslt with same shape as label
    ## save_dir: the directory for saving
    ## input_transform: the dataloader transform
    results = batch['results']
    name = batch['name']
    print(save_dir)
    
    for organ in organ_list:

        batch[ORGAN_NAME[organ-1]] = results[:,organ-1].unsqueeze(1)
        # print(batch[ORGAN_NAME[organ-1]].shape)
        post_transforms = Compose([
            Invertd(
                keys=[ORGAN_NAME[organ-1]], #, 'split_label'
                transform=input_transform,
                orig_keys=['image'],
                nearest_interp=True,
                to_tensor=True,
            ),
            SaveImaged(keys=ORGAN_NAME[organ-1], 
                    meta_keys="image_meta_dict" , 
                    output_dir=save_dir,
                    output_ext = '.nii.gz',
                    output_postfix=ORGAN_NAME[organ-1],
                    print_log = False, 
                    resample=False
            ),
        ])
        
        _ = [post_transforms(i) for i in decollate_batch(batch)]


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(1,-1)
        target = target.contiguous().view(1,-1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=NUM_CLASS, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, name, TEMPLATE):
        
        total_loss = []
        predict = F.sigmoid(predict)
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'

            else:
                template_key = name[b][0:2]

            organ_list = TEMPLATE[template_key]

            for organ in organ_list:

                dice_loss = self.dice(predict[b, organ-1], target[b, organ-1])
                total_loss.append(dice_loss)
            
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]
    
class Multi_BCELoss(nn.Module):

    def __init__(self, ignore_index=None, num_classes=NUM_CLASS, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]

            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'
        

            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]

            for organ in organ_list:

                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]

def process_ground_truth(gt_data, name):
    """
    Process ground truth data properly handling 255 as "don't care" labels
    Args:
        gt_data: Ground truth data of shape (C,H,W,D)
        name: Case name for mapping
    Returns:
        Merged ground truth with proper label values
    """
    C, H, W, D = gt_data.shape
    merged_label = np.zeros((1, H, W, D))
    template_key = get_key_2(name)
    transfer_mapping = MERGE_MAPPING_v1[template_key]
    
    # Process only valid channels (not 255)
    for src, tgt in transfer_mapping:
        channel_idx = src - 1
        channel_data = gt_data[channel_idx]
        
        # Check if this organ exists in this case
        if 255 not in np.unique(channel_data):
            # Only merge if the organ exists (not a "don't care" label)
            merged_label[0][channel_data == 1] = tgt
    
    return merged_label

def debug_label_distribution(gt_data, name):
    """
    Debug helper to understand label distribution
    Args:
        gt_data: Ground truth data
        name: Case name
    """
    print("\n=== Label Distribution Analysis ===")
    template_key = get_key_2(name)
    transfer_mapping = MERGE_MAPPING_v1[template_key]
    
    print(f"Template Key: {template_key}")
    print("Channel analysis:")
    for src, tgt in transfer_mapping:
        channel_idx = src - 1
        channel_data = gt_data[channel_idx]
        unique_vals = np.unique(channel_data)
        
        if 255 in unique_vals:
            status = "ABSENT (don't care)"
        elif 1 in unique_vals:
            status = "PRESENT"
            pixels = np.sum(channel_data == 1)
            percentage = (pixels / channel_data.size) * 100
            status += f" ({pixels} pixels, {percentage:.2f}% of volume)"
        else:
            status = "EMPTY"
            
        print(f"Organ {src} -> {tgt}: {status}")

