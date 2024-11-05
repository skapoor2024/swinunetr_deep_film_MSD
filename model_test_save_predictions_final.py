"""
To generate predictions from any model from swinunter, unetr, clip-driven and clip-deep-driven
The save_result code has some issues 
The file saved have mismatch with the orignal image meta data.
need to fix it

Here the loader will take _test2.txt files
These files dont have their ground truths
Here the files shouldn't have

"""
from types import SimpleNamespace
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

from gg_tools import  merge_label_v1, get_test_txt_loader, get_test_data_loader, dice_score, TEMPLATE, get_key, NUM_CLASS, ORGAN_NAME, organ_post_process, dice_score_np, save_result

import torch
import os
import numpy as np
import argparse

from monai.inferers import sliding_window_inference
from tqdm import tqdm

from model.SwinUNETR_DEEP_FILM import SwinUNETR_DEEP_FILM
from model.Universal_model import Universal_model

def test(model,test_loader,test_transform,args,post_process=False):

    model.eval()

    for batch in tqdm(test_loader):

        image, name, prompt = batch['image'].to(args.device), batch['name'], batch['prompt']

        with torch.no_grad():

            if(args.model_type == 'film'):
                predictor = lambda image_patch:model(image_patch,prompt)
                pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, predictor)
            else:
                pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1,model)
            pred_sigmoid = torch.nn.functional.sigmoid(pred)
        
        #now squeeze it  threshold with 0.5 ,  convert it into numpy, post_process it if needed , convert into tensor and store in batch['result']
        pred_sigmoid = torch.squeeze(pred_sigmoid)
        pred_mask = torch.where(pred_sigmoid>=0.5,1,0).to(torch.uint8).cpu().numpy()

        template_key = get_key(name[0]) #since for val_loader we have just 1 .
        organ_list = TEMPLATE[template_key]

        if post_process:
            pred_mask = organ_post_process(pred_mask,organ_list)
        
        pred_mask_merged = merge_label_v1(pred_mask,name[0])
        pred_mask_merged = pred_mask_merged.astype(np.uint8)
        #convert it into tensor and save
        batch['result'] = torch.from_numpy(np.expand_dims(pred_mask_merged,axis=0))
        
        #for path get the folder from name
        file_name = name[0].split('.')[0]
        subfold_path1, subfold_path2 = file_name.split('/')[0:2]
        #subfold_path = file_name.split('/')[1]
        save_dir = os.path.join(args.os_save_fold,subfold_path1,subfold_path2)
        #print(save_dir)
        save_result(batch,test_transform,save_dir)

def process(args):

    if args.model_type == 'film':

        from model.SwinUNETR_DEEP_FILM import SwinUNETR_DEEP_FILM

        model = SwinUNETR_DEEP_FILM(img_size=(args.roi_x, args.roi_y, args.roi_z),
                            in_channels=1,
                            out_channels=32,
                            precomputed_prompt_path=args.precomputed_prompt_path)

    elif args.model_type =='universal':

        from model.Universal_model import Universal_model
        
        model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                in_channels=1,
                out_channels=32,
                backbone=args.backbone,
                encoding=args.trans_encoding
        )

    elif args.model_type =='swinunetr':

        from monai.networks.nets import SwinUNETR

        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                in_channels=1,
                out_channels=32,
        )
    
    elif args.model_type == 'unetr':

        from monai.networks.nets import UNETR

        model = UNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32,
        )

    #create the folder to save the predicitons.
    os.makedirs(args.os_save_fold,exist_ok = True)

    ##load model weights

    checkpoint = torch.load(args.pretrain)
    store_dict = model.state_dict()
    load_dict = checkpoint['net']

    missing_wts_params = []

    #if using universal_author weights
    if args.universal_author:

        for key,value in load_dict.items():

            #print(key)
            key = '.'.join(key.split('.')[1:]) #remove module
            if 'swinViT' in key or 'encoder' in key or 'decoder' in key: #add backbone context;
                key ='.'.join(['backbone',key])
            #print(key)
            if key in store_dict.keys():
                store_dict[key]=value
            else:
                missing_wts_params.append(key)
    
    else:

        for key,value in load_dict.items():

            if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                name = '.'.join(key.split('.')[1:])
            else:
                name = '.'.join(key.split('.')[1:])
            if name in store_dict.keys():
                store_dict[name]=value
            else:
                missing_wts_params.append(name)
    
    assert len(missing_wts_params)==0,f"These weights are missing {','.join(missing_wts_params)}"

    model.load_state_dict(store_dict)

    model = model.to(args.device)

    test_loader,test_transform = get_test_txt_loader(args)
    
    test(model,test_loader,test_transform,args)

def main():

    args = SimpleNamespace(
        space_x = 1.5,
        space_y = 1.5,
        space_z = 1.5,
        roi_x = 96,
        roi_y = 96,
        roi_z = 96,
        num_samples = 2,
        data_root_path = '/blue/kgong/s.kapoor/language_guided_segmentation/CLIP-Driven-Universal-Model/',
        data_txt_path = './dataset/dataset_list/',
        batch_size = 4,
        num_workers = 8,
        a_min = -175,
        a_max = 250,
        b_min = 0.0,
        b_max = 1.0,
        dataset_list = ['PAOTtest'], #here it is used to vaidate the model
        NUM_CLASS = NUM_CLASS,
        backbone = 'swinunetr',
        trans_encoding = 'word_embedding',
        pretrain = './out/universal_total_org/epoch_400.pth',
        lr = 4e-4,
        weight_decay = 1e-5,
        precomputed_prompt_path = './pretrained_weights/embeddings_template.pkl',
        word_embedding = './pretrained_weights/txt_encoding.pth',
        dist = False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_type = 'film',
        file_name = 'paot_test_universal_postprocess.txt',
        os_save_fold = './default_prediction_space'
    )


    parser = argparse.ArgumentParser(description = 'Some arguments to take')
    parser.add_argument('--log_name', default='swinunet', help='The path resume from checkpoint')
    parser.add_argument('--precomputed_prompt_path',default='./pretrained_weights/embeddings_template_flare.pkl',help='the text embeddings to use')
    parser.add_argument('--dataset_list', nargs='+', default=['PAOTtest'], help='The dataset to be used, its txt file with location')
    parser.add_argument('--file_name',default='your_test.txt',help='where the results will be stored')
    parser.add_argument('--pretrain',default='./out/deep_film_org_setting/epoch_380.pth')
    parser.add_argument('--model_type',default='film')
    parser.add_argument('--universal_author', action='store_true', default=False)
    parser.add_argument('--os_save_fold',default = './default_prediction_space')

    parsed_args = parser.parse_args()

    args_dict = vars(parsed_args)
    for key,value in args_dict.items():
        if value is not None:
            setattr(args,key,value)

    process(args=args)

if __name__ == '__main__':

    main()

