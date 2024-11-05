"""
This file is used to generate the visualizations 
of the pretrained models
it will take the dataset list the pretrained models path the name of the pretrained model to get the visualization in the desired folder
example -> python gen_viz_model_5.py --datasetlist paottest5 --os_save_fold fold_name

To get the visualization the file should be saved in ./dataset/datasetlist/
moreover it should be saved with _val.txt suffix.

For above case paottest5_val.txt must be present
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import os
import argparse
from types import SimpleNamespace
from monai.data import DataLoader, Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch


from gg_tools import debug_label_distribution, process_ground_truth, modified_merge_label_v1, merge_label_v1, get_val_txt_loader, get_test_data_loader, dice_score, TEMPLATE, get_key, NUM_CLASS, ORGAN_NAME, organ_post_process, dice_score_np, save_result_2

def create_fixed_colormap():
    """
    Create a fixed colormap for all organ labels
    """
    # Start with transparent background
    colors = [[1, 1, 1, 0]]  # Background (transparent)
    
    # Get colors from tab20 colormap for first 20 labels
    colors.extend(plt.cm.tab20(np.linspace(0, 1, 20)))
    
    # Get additional colors from Set3 for remaining labels
    colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)))
    
    # Ensure consistent alpha for all non-background colors
    colors = np.array(colors)
    colors[1:, 3] = 0.7  # Set alpha for all non-background colors
    
    return plt.cm.colors.ListedColormap(colors)

def get_max_area_slice(gt_mask):
    """
    Find slice with maximum organ area, properly handling multi-label data
    Args:
        gt_mask: 3D numpy array with multiple labels (1,H,W,D)
    Returns:
        slice_idx: Index of the axial slice with maximum organ area
    """
    if gt_mask.shape[0] == 1:
        gt_mask = gt_mask[0]
    
    # Calculate areas for each slice
    areas = []
    for i in range(gt_mask.shape[2]):
        slice_data = gt_mask[:, :, i]
        # Count pixels that belong to any organ (any non-zero value)
        area = np.sum(slice_data > 0)
        areas.append(area)
    
    # Find slice with maximum organ area
    max_slice = np.argmax(areas)
    
    # Print diagnostic information
    print("\nSlice Selection Statistics:")
    print(f"Maximum area slice: {max_slice}")
    print(f"Area in maximum slice: {areas[max_slice]} pixels")
    print(f"Average area across slices: {np.mean(areas):.2f} pixels")
    print(f"Number of slices with organs: {np.sum(np.array(areas) > 0)}")
    
    return max_slice

def visualize_predictions(gt_volume, predictions_dict, save_path, input_image_volume, slice_idx):
    """
    Improved visualization handling multi-label data
    """
    n_models = len(predictions_dict) + 1
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
    
    custom_cmap = create_fixed_colormap()
    
    # Get input slice
    input_slice = input_image_volume[0, :, :, slice_idx]
    input_normalized = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min())
    
    # Plot ground truth
    gt_slice = gt_volume[0, :, :, slice_idx]
    axes[0].imshow(input_normalized, cmap='gray')
    axes[0].imshow(gt_slice, cmap=custom_cmap, alpha=0.7)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Plot predictions
    for idx, (model_name, pred_volume) in enumerate(predictions_dict.items(), 1):
        pred_slice = pred_volume[0, :, :, slice_idx]
        axes[idx].imshow(input_normalized, cmap='gray')
        axes[idx].imshow(pred_slice, cmap=custom_cmap, alpha=0.7)
        axes[idx].set_title(model_name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def test_all_models(args):
    """Test the models on all images in the dataset"""
    # Get data loader
    film_loader, film_transform = get_val_txt_loader(args)
    
    # Initialize models dictionary
    models = {}
    
    print("Loading models...")
    
    try:
        # Deep Film model
        print("Loading Deep Film model...")
        from model.SwinUNETR_DEEP_FILM import SwinUNETR_DEEP_FILM
        models['CLIP-Deep'] = SwinUNETR_DEEP_FILM(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32,
            precomputed_prompt_path=args.precomputed_prompt_path
        )
        
        # Universal model
        print("Loading Universal model...")
        from model.Universal_model import Universal_model
        models['Universal'] = Universal_model(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32,
            backbone=args.backbone,
            encoding=args.trans_encoding
        )
        
        # SwinUNETR
        print("Loading SwinUNETR model...")
        from monai.networks.nets import SwinUNETR
        models['SwinUNETR'] = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32
        )
        
        # UNETR
        print("Loading UNETR model...")
        from monai.networks.nets import UNETR
        models['UNETR'] = UNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32
        )
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return
    
    print("Loading model weights...")
    
    # Load pretrained weights
    for model_name, model in models.items():
        try:
            checkpoint_path = args.model_checkpoints[model_name]
            print(f"Loading weights for {model_name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            store_dict = model.state_dict()
            load_dict = checkpoint['net']
            
            if model_name == 'Universal' and args.universal_author:
                for key, value in load_dict.items():
                    key = '.'.join(key.split('.')[1:])
                    if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                        key = '.'.join(['backbone', key])
                    if key in store_dict.keys():
                        store_dict[key] = value
            else:
                for key, value in load_dict.items():
                    if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                        name = '.'.join(key.split('.')[1:])
                    else:
                        name = '.'.join(key.split('.')[1:])
                    if name in store_dict.keys():
                        store_dict[name] = value
            
            model.load_state_dict(store_dict)
            model = model.to(args.device)
            model.eval()
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading weights for {model_name}: {str(e)}")
            return
    
    # Process all images
    print("Processing all images...")
    for batch_idx, batch in enumerate(tqdm(film_loader, desc="Processing images")):
        try:
            image = batch['image'].to(args.device)
            name = batch['name']
            prompt = batch['prompt']
            
            print(f"\nProcessing image {batch_idx + 1}: {name[0]}")
            
            ground_truth = batch['post_label'][0].numpy()
            debug_label_distribution(ground_truth, name[0])  # Add this to understand label distribution
            ground_truth_merged = process_ground_truth(ground_truth, name[0])
                        
            input_image = batch['image'][0].cpu().numpy()
            
            predictions = {}
            
            with torch.no_grad():
                for model_name, model in models.items():
                    print(f"Running inference with {model_name}...")
                    if model_name == 'CLIP-Deep':
                        predictor = lambda image_patch: model(image_patch, prompt)
                        pred = sliding_window_inference(
                            image, 
                            (args.roi_x, args.roi_y, args.roi_z),
                            1,
                            predictor
                        )
                    else:
                        pred = sliding_window_inference(
                            image,
                            (args.roi_x, args.roi_y, args.roi_z),
                            1,
                            model
                        )
                    
                    pred_sigmoid = torch.squeeze(torch.sigmoid(pred))
                    pred_mask = torch.where(pred_sigmoid >= 0.5, 1, 0).cpu().numpy()
                    
                    if args.post_process:
                        template_key = get_key(name[0])
                        organ_list = TEMPLATE[template_key]
                        pred_mask = organ_post_process(pred_mask, organ_list)
                    
                    pred_mask_merged = merge_label_v1(pred_mask, name[0])
                    predictions[model_name] = pred_mask_merged.astype(np.uint8)
                    print(f"Completed {model_name} inference")
            
            # Find slice with maximum area
            max_slice = get_max_area_slice(ground_truth_merged)
            
            # Save visualization
            file_name = name[0].split('.')[0].replace('/', '_')
            save_path = os.path.join(
                args.visualization_path, 
                f'{file_name}_axial_slice_{max_slice}_comparison.png'
            )
            
            print(f"Saving visualization to {save_path}")
            visualize_predictions(
                ground_truth_merged,
                predictions,
                save_path,
                input_image,
                max_slice
            )
            
            # Calculate and print Dice scores for each model
            print("\nDice Scores:")
            for model_name, pred in predictions.items():
                dice = dice_score_np(pred, ground_truth_merged)
                print(f"{model_name}: {dice:.4f}")
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    print("\nCompleted processing all images")

def test_single_image(args):
    """Test the models on a single image first"""
    # Get the first image from the dataset
    film_loader, film_transform = get_val_txt_loader(args)
    first_batch = next(iter(film_loader))
    
    # Initialize models dictionary
    models = {}
    
    print("Loading models...")
    
    try:
        # Deep Film model
        print("Loading Deep Film model...")
        from model.SwinUNETR_DEEP_FILM import SwinUNETR_DEEP_FILM
        models['CLIP-Deep'] = SwinUNETR_DEEP_FILM(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32,
            precomputed_prompt_path=args.precomputed_prompt_path
        )
        
        # Universal model
        print("Loading Universal model...")
        from model.Universal_model import Universal_model
        models['Universal'] = Universal_model(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32,
            backbone=args.backbone,
            encoding=args.trans_encoding
        )
        
        # SwinUNETR
        print("Loading SwinUNETR model...")
        from monai.networks.nets import SwinUNETR
        models['SwinUNETR'] = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32
        )
        
        # UNETR
        print("Loading UNETR model...")
        from monai.networks.nets import UNETR
        models['UNETR'] = UNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=32
        )
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return
    
    print("Loading model weights...")
    
    # Load pretrained weights
    for model_name, model in models.items():
        try:
            checkpoint_path = args.model_checkpoints[model_name]
            print(f"Loading weights for {model_name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            store_dict = model.state_dict()
            load_dict = checkpoint['net']
            
            if model_name == 'Universal' and args.universal_author:
                for key, value in load_dict.items():
                    key = '.'.join(key.split('.')[1:])
                    if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                        key = '.'.join(['backbone', key])
                    if key in store_dict.keys():
                        store_dict[key] = value
            else:
                for key, value in load_dict.items():
                    if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                        name = '.'.join(key.split('.')[1:])
                    else:
                        name = '.'.join(key.split('.')[1:])
                    if name in store_dict.keys():
                        store_dict[name] = value
            
            model.load_state_dict(store_dict)
            model = model.to(args.device)
            model.eval()
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading weights for {model_name}: {str(e)}")
            return
    
    print("Processing single image...")
    
    # Process the single image
    try:
        image = first_batch['image'].to(args.device)
        name = first_batch['name']
        prompt = first_batch['prompt']
        
        print(f"Processing image: {name[0]}")
        
        print(first_batch['post_label'].shape)
        ground_truth = first_batch['post_label'][0].numpy()
        debug_label_distribution(ground_truth, name[0])  # Add this to understand label distribution
        ground_truth_merged = process_ground_truth(ground_truth, name[0])
        
        # Store original input image for visualization
        input_image = first_batch['image'][0].cpu().numpy()
        
        predictions = {}
        
        with torch.no_grad():
            for model_name, model in models.items():
                print(f"Running inference with {model_name}...")
                if model_name == 'CLIP-Deep':
                    predictor = lambda image_patch: model(image_patch, prompt)
                    pred = sliding_window_inference(
                        image, 
                        (args.roi_x, args.roi_y, args.roi_z),
                        1,
                        predictor
                    )
                else:
                    pred = sliding_window_inference(
                        image,
                        (args.roi_x, args.roi_y, args.roi_z),
                        1,
                        model
                    )
                
                pred_sigmoid = torch.squeeze(torch.sigmoid(pred))
                pred_mask = torch.where(pred_sigmoid >= 0.5, 1, 0).cpu().numpy()
                
                if args.post_process:
                    template_key = get_key(name[0])
                    organ_list = TEMPLATE[template_key]
                    pred_mask = organ_post_process(pred_mask, organ_list)
                
                pred_mask_merged = merge_label_v1(pred_mask, name[0])
                predictions[model_name] = pred_mask_merged.astype(np.uint8)
                print(f"Completed {model_name} inference")
                
        max_slice = get_max_area_slice(ground_truth_merged)
        
        # Save visualization
        file_name = name[0].split('.')[0].replace('/', '_')
        save_path = os.path.join(
            args.visualization_path, 
            f'{file_name}_axial_slice_{max_slice}_comparison.png'
        )
        
        print(f"Saving visualization to {save_path}")
        visualize_predictions(
            ground_truth_merged,
            predictions,
            save_path,
            input_image,
            max_slice
        )
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise e

    

def process(args):
    # Add new arguments needed for multi-model comparison
    args.visualization_path = os.path.join(args.os_save_fold, 'visualizations')
    args.model_checkpoints = {
        'CLIP-Deep': args.deep_film_checkpoint,
        'Universal': args.universal_checkpoint,
        'SwinUNETR': args.swinunetr_checkpoint,
        'UNETR': args.unetr_checkpoint
    }
    
    # Create visualization directory
    os.makedirs(args.visualization_path, exist_ok=True)
    
    # Test with single image first
    print("Testing with single image...")
    test_single_image(args)
    
    # If successful, proceed with all images
    user_input = input("Continue with all images? (y/n): ")
    if user_input.lower() == 'y':
        print("Processing all images...")
        test_all_models(args)
    else:
        print("Exiting after single image test")

def main():
    # Define default arguments
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
        num_workers = 1,
        a_min = -175,
        a_max = 250,
        b_min = 0.0,
        b_max = 1.0,
        dataset_list = ['PAOTtest'], #here it is used to validate the model
        NUM_CLASS = 32,  # Make sure this matches your actual number of classes
        backbone = 'swinunetr',
        trans_encoding = 'word_embedding',
        lr = 4e-4,
        weight_decay = 1e-5,
        precomputed_prompt_path = './pretrained_weights/embeddings_template.pkl',
        word_embedding = './pretrained_weights/txt_encoding.pth',
        dist = False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_type = 'film',
        file_name = 'paot_test_universal_postprocess.txt',
        os_save_fold = './default_prediction_space',
        deep_film_checkpoint = './out/deep_film_org_setting/epoch_380.pth',
        universal_checkpoint = './out/universal_total_org/epoch_380.pth',
        swinunetr_checkpoint = './out/swinunetr_monai/epoch_120.pth',
        unetr_checkpoint = './out/unetr_monai/epoch_120.pth',
        universal_author = False,
        post_process = True
    )

    # Set up argument parser for command line overrides
    parser = argparse.ArgumentParser(description='Multi-model comparison for medical image segmentation')
    
    parser.add_argument('--precomputed_prompt_path', 
                       default='./pretrained_weights/embeddings_template_flare.pkl',
                       help='the text embeddings to use')
    
    parser.add_argument('--dataset_list', 
                       nargs='+', 
                       default=['PAOTtest'], 
                       help='The dataset to be used, its txt file with location')
    
    parser.add_argument('--deep_film_checkpoint',
                       default='./out/deep_film_org_setting/epoch_380.pth',
                       help='Path to Deep Film model checkpoint')
    
    parser.add_argument('--universal_checkpoint',
                       default='./out/universal_total_org/epoch_380.pth',
                       help='Path to Universal model checkpoint')
    
    parser.add_argument('--swinunetr_checkpoint',
                       default='./out/swinunetr_monai/epoch_120.pth',
                       help='Path to SwinUNETR model checkpoint')
    
    parser.add_argument('--unetr_checkpoint',
                       default='./out/unetr_monai/epoch_120.pth',
                       help='Path to UNETR model checkpoint')
    
    parser.add_argument('--universal_author', 
                       action='store_true',
                       default=False,
                       help='Use Universal author weight loading method')
    
    parser.add_argument('--os_save_fold',
                       default='./default_prediction_space',
                       help='Directory to save output predictions')
    
    parser.add_argument('--post_process',
                       action='store_true',
                       default=True,
                       help='Apply post-processing to predictions')

    # Parse command line arguments
    parsed_args = parser.parse_args()

    # Update default arguments with any command line overrides
    args_dict = vars(parsed_args)
    for key, value in args_dict.items():
        if value is not None:
            setattr(args, key, value)

    # Print configuration
    print("\nRunning with configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("\n")

    # Run the main process
    process(args=args)

if __name__ == '__main__':
    main()
