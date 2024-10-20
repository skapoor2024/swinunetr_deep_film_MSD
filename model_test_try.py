from types import SimpleNamespace
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

from gg_tools import get_train_val_data_loader, get_train_val_txt_loader, dice_score, TEMPLATE, get_key_2, NUM_CLASS, ORGAN_NAME, organ_post_process, dice_score_np

import torch
import os
import numpy as np
import argparse

from monai.inferers import sliding_window_inference
from tqdm import tqdm

from model.SwinUNETR_DEEP_FILM import SwinUNETR_DEEP_FILM
from model.Universal_model import Universal_model

def validation_postprocess(model,val_loader,args,post_process=True):

    model.eval()

    dice_list = {key: torch.zeros(2,NUM_CLASS).to(args.device) for key in TEMPLATE.keys()}

    for batch in tqdm(val_loader):

        
        if(args.model_type == 'film'):
            image, label, name, prompt = batch['image'].to(args.device), batch['post_label'], batch['name'], batch['prompt']
        else:
            image, label, name = batch['image'].to(args.device), batch['post_label'], batch['name']

        with torch.no_grad():

            if(args.model_type == 'film'):
                predictor = lambda image_patch:model(image_patch,prompt)
                pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, predictor)
            else:
                pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model)
            pred_sigmoid = torch.nn.functional.sigmoid(pred)
        
        template_key = get_key_2(name[0]) #since for val_loader we have just 1 .
        organ_list = TEMPLATE[template_key]

        pred_sigmoid = torch.squeeze(pred_sigmoid)
        pred_sigmoid = torch.where(pred_sigmoid>0.5,1.,0.)
        pred_mask = pred_sigmoid.cpu().numpy()

        if post_process:
            post_processed_mask = organ_post_process(pred_mask,organ_list)
        else:
            post_processed_mask = pred_mask

        label = np.array(label)
        label = np.squeeze(label)

        for organ in organ_list:
            dice_organ = dice_score_np(post_processed_mask[organ-1,:,:,:], label[organ-1,:,:,:])
            dice_list[template_key][0][organ-1] += dice_organ
            dice_list[template_key][1][organ-1] += 1
    
    avg_organ_dice = np.zeros((2,NUM_CLASS))

    with open(args.file_name, 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                avg_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                avg_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, '%(ORGAN_NAME[i], avg_organ_dice[0][i] / avg_organ_dice[1][i])
        f.write(content)
        f.write('\n')

    return avg_organ_dice

def process(args):

    if args.model_type == 'film':

        model = SwinUNETR_DEEP_FILM(img_size=(args.roi_x, args.roi_y, args.roi_z),
                            in_channels=1,
                            out_channels=32,
                            precomputed_prompt_path=args.precomputed_prompt_path)
    else:
        
        clip_model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                in_channels=1,
                out_channels=32,
                backbone=args.backbone,
                encoding=args.trans_encoding
        )

    ##load model weights
    
    checkpoint = torch.load(args.pretrain)
    store_dict = model.state_dict()
    load_dict = checkpoint['net']

    missing_wts_params = []

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

    if args.model_type == 'film':

        train_loader, val_loader,train_sampler, val_sampler = get_train_val_txt_loader(args)
    
    else:
        train_loader, val_loader,train_sampler, val_sampler = get_train_val_data_loader(args)

    
    avg_organ_dice = validation_postprocess(model,val_loader,args)

    print(avg_organ_dice)

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
        os_save_fold = './not_required'
    )


    parser = argparse.ArgumentParser(description = 'Some arguments to take')
    parser.add_argument('--log_name', default='swinunet', help='The path resume from checkpoint')
    parser.add_argument('--precomputed_prompt_path',default='./pretrained_weights/embeddings_template_flare.pkl',help='the text embeddings to use')
    parser.add_argument('--dataset_list', nargs='+', default=['PAOTtest'], help='The dataset to be used, its txt file with location')
    parser.add_argument('--file_name',default='your_test.txt',help='where the results will be stored')
    parser.add_argument('--pretrain',default='./out/deep_film_org_setting/epoch_380.pth')
    parser.add_argument('--model_type',default='film')

    parsed_args = parser.parse_args()

    args_dict = vars(parsed_args)
    for key,value in args_dict.items():
        if value is not None:
            setattr(args,key,value)

    process(args=args)

if __name__ == '__main__':

    main()

