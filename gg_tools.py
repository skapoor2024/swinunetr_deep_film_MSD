""""
Rewriting the data loader , loss and other functions here.
Rewriting whole script lol
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
)

import h5py
import numpy as np

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

DEFAULT_POST_FIX = PostFix.meta()

ORGAN_NAME = ['Spleen','Liver','Pancreas','Hepatic Vessel','Liver Tumor','Pancreas Tumor','Hepatic Vessel Tumor','Lung Tumor','Colon Tumor']

TEMPLATE = {
    '10_03': [2, 5], # post process
    '10_06': [8],
    '10_07': [3, 6], # post process
    '10_08': [4, 7], # post process
    '10_09': [1],
    '10_10': [9],
}

NUM_CLASS = 9

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
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
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]

            for organ in organ_list:

                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]

