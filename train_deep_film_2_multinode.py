import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import socket

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.SwinUNETR_DEEP_FILM_2 import SwinUNETR_DEEP_FILM
from dataset.dataloader import get_loader_with_text
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


torch.multiprocessing.set_sharing_strategy('file_system')

def setup_distributed(args):
    """Initialize distributed training"""
    if args.dist:
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.global_rank
        )
        
        # Log distributed training information
        print(f"[Rank {args.global_rank}/{args.world_size}] Initialized process group")
        print(f"[Rank {args.global_rank}] Local Rank: {args.local_rank}")
        print(f"[Rank {args.global_rank}] Node Rank: {args.node_rank}")
        print(f"[Rank {args.global_rank}] Master: {args.master_addr}:{args.master_port}")

def cleanup():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_model_weights(model, checkpoint_path,is_resume=False):

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    store_dict = model.state_dict()
    load_dict = checkpoint['net']
    missing_wts_params = []

    # Handle standard weights format
    for key, value in load_dict.items():

        if any(x in key for x in ['swinViT', 'encoder', 'decoder']):
            name = '.'.join(key.split('.')[1:])
        else:
            name = '.'.join(key.split('.')[1:])
            
        if name in store_dict:
            store_dict[name] = value
        else:
            missing_wts_params.append(name)

    # Verify all weights were loaded
    if missing_wts_params:
        raise AssertionError(f"These weights are missing: {', '.join(missing_wts_params)}")
    
    # Load weights into model
    model.load_state_dict(store_dict)
    
    if is_resume:
        # Return additional states for resume scenario
        return model, {
            'optimizer': checkpoint.get('optimizer'),
            'scheduler': checkpoint.get('scheduler'),
            'epoch': checkpoint.get('epoch')
        }
    else:
        return model, None

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name, prompt = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name'], batch['prompt']
        logit_map = model(x,prompt)

        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

def process(args):

    # Calculate ranks for distributed training
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.node_rank = int(os.environ.get("SLURM_NODEID", 0))
    args.global_rank = args.node_rank * args.gpus_per_node + args.local_rank
    
    # Setup distributed training
    if args.dist:
        setup_distributed(args)
    
    # Set device
    args.device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(args.device)

    # Model initialization
    model = SwinUNETR_DEEP_FILM(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=NUM_CLASS,
                        precomputed_prompt_path=args.precomputed_prompt_path)

    #Load pre-trained weights
    if args.pretrain is not None:
        model,_ = load_model_weights(
            model = model,
            checkpoint_path = args.pretrain,
            is_resume = False
        )

    model.to(args.device)

    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:

        model, states = load_model_weights(
            model=model,
            checkpoint_path=args.resume,
            is_resume=True
        )

        if states:

            optimizer.load_state_dict(states['optimizer'])
            scheduler.load_state_dict(states['scheduler'])
            args.epoch = states['epoch']
        
        print('success resume from ', args.resume)

    model.train()
    if args.dist:
        model = DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader_with_text(args) #make edits to loader, add text prompts to it.

    if args.global_rank == 0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to', 'out/' + args.log_name)

    while args.epoch < args.max_epoch:

        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)

        if args.global_rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', np.array(scheduler.get_lr()), args.epoch)

            # Save checkpoint
            if args.epoch % args.store_num == 0 and args.epoch != 0:
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
                os.makedirs('out/' + args.log_name, exist_ok=True)
                torch.save(checkpoint, f'out/{args.log_name}/epoch_{args.epoch}.pth')
                print('Saved model checkpoint')

        args.epoch += 1

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int, default= int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='deep_film_org_setting', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=50, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='/blue/kgong/s.kapoor/language_guided_segmentation/CLIP-Driven-Universal-Model_org/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--precomputed_prompt_path',default='./pretrained_weights/embeddings_template_flare.pkl')

    # for distributed training
    parser.add_argument('--world_size', type=int, default=int(os.environ.get("WORLD_SIZE", 1)),
                       help='Total number of processes to run')
    parser.add_argument('--master_addr', type=str, default=os.environ.get("MASTER_ADDR", "localhost"),
                       help='Master node address')
    parser.add_argument('--master_port', type=str, default=os.environ.get("MASTER_PORT", "29500"),
                       help='Master node port')
    parser.add_argument('--gpus_per_node', type=int, default=1,
                       help='Number of GPUs per node')
    
    args = parser.parse_args()
    
    try:
        process(args=args)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        cleanup()
        raise e

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()


