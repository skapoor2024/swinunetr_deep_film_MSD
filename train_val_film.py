import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

#load model
from model.SwinUNETR_FLIM import SwinUNETR_FILM

from torch.utils.tensorboard import SummaryWriter

from monai.inferers import sliding_window_inference

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from types import SimpleNamespace

from gg_tools import dice_score, TEMPLATE, get_key, NUM_CLASS, ORGAN_NAME, get_train_val_txt_loader, DiceLoss, Multi_BCELoss

torch.multiprocessing.set_sharing_strategy('file_system')

def aggregate_distributed_losses(args,loss_dice,loss_bce,len_iter):

    total_dice_loss = loss_dice * len_iter
    total_bce_loss = loss_bce * len_iter

    if args.dist:

        tensors = {
            'dice_loss': torch.tensor(total_dice_loss,devcice = args.device),
            'bce_loss': torch.tensor(total_bce_loss,device = args.device),
            'samples': torch.tensor(len_iter,args.device)
        }

        for tensor in tensors.values():
            dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
        
        avg_dice_loss = tensors['dice_loss']/tensors['samples']
        avg_bce_loss = tensors['bce_loss']/tensors['samples']

    else:

        avg_dice_loss = total_dice_loss/len_iter
        avg_bce_loss = total_bce_loss/len_iter
    
    return avg_dice_loss.item(),avg_bce_loss.item()

#training process

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):

    model.train()

    loss_bce_sum = torch.tensor(0.0).to(args.device)
    loss_dice_sum = torch.tensor(0.0).to(args.device)
    total_steps = torch.tensor(0).to(args.device)

    epoch_iterator = tqdm(
        train_loader,
        desc=f"Epoch {args.epoch}: Training",
        disable=args.local_rank!=0,
        dynamic_ncols=True
    )

    for step, batch in enumerate(epoch_iterator):

        x = batch["image"].to(args.device)
        y = batch["post_label"].to(args.device).float()
        prompt = batch['prompt']
        name = batch['name']

        logit_map = model(x,prompt)
        term_seg_Dice = loss_seg_DICE(logit_map, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE(logit_map, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_bce_sum += term_seg_BCE.item()
        loss_dice_sum += term_seg_Dice.item()
        total_steps += 1

        if args.local_rank==0:
            epoch_iterator.set_description(
                f"Epoch {args.epoch}: Training ({step+1}/{len(train_loader)}) "
                f"(dice_loss={term_seg_Dice.item():.5f}, bce_loss={term_seg_BCE.item():.5f})"
            )

        torch.cuda.empty_cache()

    # Synchronize the sum of losses and total steps across all processes
    dist.all_reduce(loss_bce_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_dice_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)

    # Calculate average losses
    avg_bce_loss = loss_bce_sum.item() / total_steps.item()
    avg_dice_loss = loss_dice_sum.item() / total_steps.item()

    if args.local_rank==0:
        print(f'Epoch {args.epoch}: avg_dice_loss={avg_dice_loss:.5f}, '
              f'avg_bce_loss={avg_bce_loss:.5f} (across all GPUs)')

    return avg_dice_loss, avg_bce_loss, total_steps.item()

#validation process 
def distributed_validation(model, ValLoader,args):

    model.eval()

    dice_list = {key: torch.zeros(2,NUM_CLASS).to(args.device) for key in TEMPLATE.keys()}

    for batch in tqdm(ValLoader,disable = args.local_rank!=0):

        image, label, name, prompt = batch['image'].to(args.device), batch['post_label'], batch['name'], batch['prompt']

        with torch.no_grad():
            predictor = lambda image_patch:model(image_patch,prompt)
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, predictor)
            pred_sigmoid = F.sigmoid(pred)

        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())[0]
                dice_list[template_key][0][organ-1] += dice_organ.item()
                dice_list[template_key][1][organ-1] += 1

    #accumulate resutls across all GPUs
    for key in TEMPLATE.keys():
        dist.all_reduce(dice_list[key],op = dist.ReduceOp.SUM)

    dice_list = {key: value.cpu().numpy() for key,value in dice_list.items()}

    #calculate average dice scores

    avg_organ_dice = np.zeros((2,NUM_CLASS))

    if args.local_rank == 0:

        with open('out/'+args.log_name+f'/val_{args.epoch}.txt', 'w') as f:
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

def validation(model, ValLoader, args):

    model.eval()
   
    dice_list = {}

    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count

    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model)
            pred_sigmoid = F.sigmoid(pred)

        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())[0]
                dice_list[template_key][0][organ-1] += dice_organ.item()
                dice_list[template_key][1][organ-1] += 1
        #if(index == 10): break

    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open('out/'+args.log_name+f'/val_{args.epoch}.txt', 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
        f.write(content)
        f.write('\n')

    return ave_organ_dice

def process(args):
    
    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
        print(args.local_rank)

    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)
    print(args.device)

    # Model initialization
    model = SwinUNETR_FILM(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=NUM_CLASS,
                        precomputed_prompt_path=args.precomputed_prompt_path)
    
    if args.pretrain:
        model.load_params(torch.load(args.pretrain)["state_dict"])
    
    model.to(args.device)
    
    if args.dist:
        model = DDP(model, device_ids=[args.device])
    # criterion and optimizer    
    loss_seg_DICE = DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, val_loader,train_sampler, val_sampler = get_train_val_txt_loader(args)

    if rank==0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to ', 'out/' + args.log_name)
        print('training started')
    
    while args.epoch < args.max_epoch:

        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
            val_sampler.set_epoch(args.epoch)
        
        scheduler.step()

        avg_loss_dice, avg_loss_bce, len_iter = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
                
        avg_organ_dice_val = distributed_validation(model,val_loader,args) #getting average organ dice loss in validation set.

        if args.local_rank == 0:

            writer.add_scalar('train_dice_loss', avg_loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', avg_loss_bce, args.epoch)
            writer.add_scalar('lr', np.array(scheduler.get_lr()), args.epoch)        

            for i in range(avg_organ_dice_val.shape[1]):

                writer.add_scalar(f'Dice Score Class {ORGAN_NAME[i]}',avg_organ_dice_val[0][i] / avg_organ_dice_val[1][i],args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and args.local_rank==0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir('out/' + args.log_name):
                os.mkdir('out/' + args.log_name)
            torch.save(checkpoint, 'out/' + args.log_name + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success')

        args.epoch += 1

def main():

    args = SimpleNamespace(
        space_x = 1.5,
        space_y = 1.5,
        space_z = 1.5,
        roi_x = 96,
        roi_y = 96,
        roi_z = 96,
        num_samples = 2,
        data_root_path = '/blue/kgong/s.kapoor/language_guided_segmentation/CLIP-Driven-Universal-Model_MSD_only/',
        data_txt_path = './dataset/dataset_list/',
        precomputed_prompt_path = './pretrained_weights/embeddings_template.pkl',
        batch_size = 4,
        num_workers = 8,
        a_min = -175,
        a_max = 250,
        b_min = 0.0,
        b_max = 1.0,
        dataset_list = ['PAOT_10_inner'],
        NUM_CLASS = 9,
        backbone = 'swinunetr',
        trans_encoding = 'word_embedding',
        #pretrain = './out/swinunetr_dist_msd/epoch_20.pth',
        pretrain = None,
        lr = 1e-4,
        weight_decay = 1e-5,
        dist = True,
        max_epoch = 500,
        store_num = 10,
        warmup_epoch = 10,
        epoch = 0,
        local_rank = int(os.environ['LOCAL_RANK']),
        device = None,
        resume = None
    )

    #args to parse are as follows:
    parser = argparse.ArgumentParser(description = 'Some arguments to take')
    parser.add_argument('--log_name', default='swinunet', help='The path resume from checkpoint')

    parsed_args = parser.parse_args()

    args_dict = vars(parsed_args)
    for key,value in args_dict.items():
        if value is not None:
            setattr(args,key,value)

    process(args=args)

if __name__ == "__main__":

    main()

# torchrun --nproc_per_node=4 --master_port=1234 train_val_dist.py --dist True --backbone swinunetr --log_name swinunetr_dist_model --batch_size 3 


