#!/usr/bin/env python3
"""
Block-wise PCK Evaluation for Vision Transformers

This script evaluates the Percentage of Correct Keypoints (PCK) metric across
different transformer blocks of vision transformers (SAM, DINOv2, DINOv3). 
It extracts features cumulatively through blocks (0 to N) to identify the 
optimal block depth for semantic correspondence tasks.

Supported Models:
    - SAM: vit_b, vit_l, vit_h (1024x1024 resolution)
    - DINOv2: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14 (518x518 resolution)
    - DINOv3: dinov3_vits16 (512x512 resolution)

Usage Examples:
    # SAM model (requires weights)
    python evaluate_blocks.py \
        --model-type vit_b \
        --weights-path path/to/sam_vit_b.pth \
        --data-root data/SPair-71k \
        --split test \
        --block-start 6 \
        --block-end 11 \
        --output-dir evaluations/block_sweep
    
    # DINOv2 model (auto-downloads if no weights provided)
    python evaluate_blocks.py \
        --model-type dinov2_vits14 \
        --data-root data/SPair-71k \
        --split test \
        --output-dir evaluations/dinov2_block_sweep
    
    # DINOv3 model
    python evaluate_blocks.py \
        --model-type dinov3_vits16 \
        --data-root data/SPair-71k \
        --split test \
        --output-dir evaluations/dinov3_block_sweep
"""

import argparse
import os
import csv
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.models import SAMAdapter, DINOv2Adapter, DINOv3Adapter
from src.spair_dataset import SPair71kPairs
from src.new_loss import predict_keypoints, predict_keypoints_window, denormalize_predictions
from src.pck import compute_pck_from_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SAM PCK performance across different transformer blocks'
    )
    
    # Model configuration
    parser.add_argument('--model-type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h', 
                               'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                               'dinov3_vits16'],
                       help='Model variant (SAM: vit_b/vit_l/vit_h, DINOv2: dinov2_vits14/vitb14/vitl14/vitg14, DINOv3: dinov3_vits16)')
    parser.add_argument('--weights-path', type=str, required=False,
                       help='Path to model checkpoint (.pth file). Required for SAM, optional for DINO models (will download if not provided)')
    
    # Data configuration
    parser.add_argument('--data-root', type=str, default='data/SPair-71k',
                       help='Path to SPair-71k dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['test', 'val', 'trn'],
                       help='Dataset split to evaluate')
    
    # Block range configuration
    parser.add_argument('--block-start', type=int, default=None,
                       help='First block index to evaluate (supports negative indexing). '
                            'Default: -6 (6th block from end)')
    parser.add_argument('--block-end', type=int, default=None,
                       help='Last block index to evaluate (supports negative indexing). '
                            'Default: -1 (last block)')
    
    # Evaluation settings
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='PCK threshold (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Evaluation batch size (current implementation uses 1)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Limit number of samples to evaluate (for quick testing)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='evaluations/block_sweep',
                       help='Directory for results and visualizations')
    
    return parser.parse_args()


def evaluate_single_block(model, dataloader, block_idx, alpha, standard_size, device, num_samples=None):
    """
    Evaluate PCK for features extracted through a specific block depth.
    Uses cumulative extraction (blocks 0 through block_idx).
    
    Args:
        model: SAMAdapter instance
        dataloader: DataLoader for evaluation pairs
        block_idx: Block index to evaluate through (inclusive)
        alpha: PCK threshold
        standard_size: Image size for feature extraction
        device: torch device
        num_samples: Optional limit on number of samples
    
    Returns:
        dict: Results containing PCK metrics and per-sample details
    """
    model.model.eval()
    
    print(f"Evaluating through Block {block_idx} (cumulative: blocks 0-{block_idx})...")
    
    results = {
        'pck_global_samples': [],
        'pck_window_samples': [],
        'errors_global': [],
        'errors_window': []
    }
    
    
    sample_count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Block {block_idx}")):

        if num_samples and sample_count >= num_samples:
            break
        
        # Unpack batch (custom collate returns lists)
        src_name = batch['src_name'][0]
        trg_name = batch['trg_name'][0]
        src_kps_raw = torch.tensor(batch['src_kps'][0], dtype=torch.float32)
        trg_kps_raw = torch.tensor(batch['trg_kps'][0], dtype=torch.float32)
        trg_bbox = torch.tensor(batch['trg_bndbox'][0], dtype=torch.float32)

        
        # Load and preprocess images
        src_img_path = dataloader.dataset.get_image_path(src_name)
        trg_img_path = dataloader.dataset.get_image_path(trg_name)
        
        src_tensor = model.preprocess_image(str(src_img_path), target_size=(standard_size, standard_size))
        trg_tensor = model.preprocess_image(str(trg_img_path), target_size=(standard_size, standard_size))
        
        # Get original image sizes from batch
        src_orig_size = batch['src_imsize'][0]
        trg_orig_size = batch['trg_imsize'][0]
        src_w, src_h = int(src_orig_size[0]), int(src_orig_size[1])
        trg_w, trg_h = int(trg_orig_size[0]), int(trg_orig_size[1])

        
        # Extract features through specified block (cumulative)
        with torch.no_grad():
            src_block_feat = model.extract_features_from_block(src_tensor, block_idx)
            trg_block_feat = model.extract_features_from_block(trg_tensor, block_idx)
            
            # Apply neck to get final features
            feat1 = model.forward_from_block_features(src_block_feat)
            feat2 = model.forward_from_block_features(trg_block_feat)
        
        # Scale source keypoints from original size to standard size
        src_kps = src_kps_raw.clone().float()
        src_kps[:, 0] *= (standard_size / src_w)
        src_kps[:, 1] *= (standard_size / src_h)
        src_input = src_kps.to(device).unsqueeze(0)

        
        # Predict keypoints (both methods)
        with torch.no_grad():
            pred_global_norm = predict_keypoints(
                feat1, feat2, src_input,
                src_img_size=(standard_size, standard_size),
                temperature=0.1
            )
            pred_window_norm = predict_keypoints_window(
                feat1, feat2, src_input,
                src_img_size=(standard_size, standard_size),
                temperature=0.1
            )
        
        # Denormalize predictions to original target image size
        pred_global_orig = denormalize_predictions(pred_global_norm, (trg_w, trg_h)).squeeze(0).cpu()
        pred_window_orig = denormalize_predictions(pred_window_norm, (trg_w, trg_h)).squeeze(0).cpu()

        
        # Compute PCK
        trg_kps = trg_kps_raw.float()
        dist_global = torch.norm(pred_global_orig - trg_kps, dim=-1)
        dist_window = torch.norm(pred_window_orig - trg_kps, dim=-1)
        
        bbox_size = max(trg_bbox[2] - trg_bbox[0], trg_bbox[3] - trg_bbox[1]).item()
        threshold = alpha * bbox_size
        
        # Filter visible keypoints
        valid_mask = (trg_kps[:, 0] > 0) & (trg_kps[:, 1] > 0)
        
        if valid_mask.sum() > 0:
            pck_global = (dist_global[valid_mask] <= threshold).float().mean().item()
            pck_window = (dist_window[valid_mask] <= threshold).float().mean().item()
            err_global = dist_global[valid_mask].mean().item()
            err_window = dist_window[valid_mask].mean().item()
            
            results['pck_global_samples'].append(pck_global)
            results['pck_window_samples'].append(pck_window)
            results['errors_global'].append(err_global)
            results['errors_window'].append(err_window)
        
        sample_count += 1
    
    # Aggregate results
    results['pck_global_mean'] = np.mean(results['pck_global_samples']) if results['pck_global_samples'] else 0.0
    results['pck_window_mean'] = np.mean(results['pck_window_samples']) if results['pck_window_samples'] else 0.0
    results['error_global_mean'] = np.mean(results['errors_global']) if results['errors_global'] else 0.0
    results['error_window_mean'] = np.mean(results['errors_window']) if results['errors_window'] else 0.0
    results['num_samples'] = len(results['pck_global_samples'])
    
    return results


def plot_block_comparison(results_df, save_path):
    """
    Create visualization comparing PCK across different block depths.
    
    Args:
        results_df: DataFrame with columns ['block_idx', 'pck_global', 'pck_window', 'error']
        save_path: Output path for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: PCK scores
    ax1.plot(results_df['block_idx'], results_df['pck_global'], 
             marker='o', label='Global PCK', linewidth=2, markersize=8)
    ax1.plot(results_df['block_idx'], results_df['pck_window'], 
             marker='s', label='Window PCK', linewidth=2, markersize=8)
    ax1.set_xlabel('Block Index', fontsize=12)
    ax1.set_ylabel('PCK (%)', fontsize=12)
    ax1.set_title('PCK Performance Across Block Depths', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Highlight best performing block
    best_idx = results_df['pck_window'].idxmax()
    best_block = results_df.loc[best_idx, 'block_idx']
    best_pck = results_df.loc[best_idx, 'pck_window']
    ax1.axvline(x=best_block, color='green', linestyle='--', alpha=0.5, label=f'Best: Block {best_block}')
    ax1.legend(fontsize=10)
    
    # Plot 2: Average error
    ax2.plot(results_df['block_idx'], results_df['error_global'], 
             marker='o', label='Global Error', linewidth=2, markersize=8, color='coral')
    ax2.plot(results_df['block_idx'], results_df['error_window'], 
             marker='s', label='Window Error', linewidth=2, markersize=8, color='lightblue')
    ax2.set_xlabel('Block Index', fontsize=12)
    ax2.set_ylabel('Average Error (pixels)', fontsize=12)
    ax2.set_title('Prediction Error Across Block Depths', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model family and initialize appropriate adapter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model_type.startswith('dinov2'):
        print(f"Loading DINOv2 {args.model_type}...")
        model = DINOv2Adapter(
            model_name=args.model_type,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=0,  # Keep all blocks frozen for evaluation
        )
    elif args.model_type.startswith('dinov3'):
        print(f"Loading DINOv3 {args.model_type}...")
        model = DINOv3Adapter(
            model_name=args.model_type,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=0,  # Keep all blocks frozen for evaluation
        )
    else:  # SAM models (vit_b, vit_l, vit_h)
        print(f"Loading SAM {args.model_type}...")
        if not args.weights_path:
            raise ValueError("--weights-path is required for SAM models")
        model = SAMAdapter(
            model_name=args.model_type,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=0,  # Keep all blocks frozen for evaluation
        )
    
    num_blocks = len(model.model.blocks)
    print(f"Model has {num_blocks} transformer blocks")
    print(f"Using image size: {model.standard_size}x{model.standard_size}")
    
    # Determine block range
    block_start = args.block_start if args.block_start is not None else -6
    block_end = args.block_end if args.block_end is not None else -1
    
    # Convert negative indices
    if block_start < 0:
        block_start = num_blocks + block_start
    if block_end < 0:
        block_end = num_blocks + block_end
    
    block_range = range(block_start, block_end + 1)
    print(f"Evaluating blocks: {list(block_range)}")
    
    # Load dataset
    print(f"Loading {args.split} split from {args.data_root}...")
    # Note: SPair71kPairs adds 'SPair-71k' to the root, so we pass the parent directory
    dataset_root = Path(args.data_root).parent if 'SPair-71k' in args.data_root else args.data_root
    dataset = SPair71kPairs(root=dataset_root, split=args.split)
    
    # Create dataloader (batch_size=1 for simplicity)
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate to add dataset root to batch."""
        batch_dict = {}
        for key in batch[0].keys():
            if key in ['src_img', 'trg_img', 'src_segmentation', 'trg_segmentation']:
                # Skip image data - we'll load directly
                continue
            batch_dict[key] = [item[key] for item in batch]
        return batch_dict
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    
    print(f"Dataset size: {len(dataset)} pairs")
    if args.num_samples:
        print(f"Limiting evaluation to {args.num_samples} samples")
    
    # Run block-wise evaluation
    print(f"\n{'='*60}")
    print(f"BLOCK-WISE PCK EVALUATION")
    print(f"{'='*60}\n")
    
    all_results = []
    
    # Evaluate each block
    for block_idx in block_range:
        result = evaluate_single_block(
            model, dataloader, block_idx, 
            args.alpha, model.standard_size, device,
            num_samples=args.num_samples
        )
            
        all_results.append({
            'block_idx': block_idx,
            'pck_global': result['pck_global_mean'],
            'pck_window': result['pck_window_mean'],
            'error_global': result['error_global_mean'],
            'error_window': result['error_window_mean'],
            'num_samples': result['num_samples']
        })
            
        print(f"Block {block_idx}: PCK-Window={result['pck_window_mean']:.2%}, "
              f"PCK-Global={result['pck_global_mean']:.2%}, "
              f"Error={result['error_window_mean']:.2f}px")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, 'block_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate visualization
    plot_path = os.path.join(args.output_dir, 'block_comparison.png')
    plot_block_comparison(results_df, plot_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    best_idx = results_df['pck_window'].idxmax()
    best_block = results_df.loc[best_idx, 'block_idx']
    best_pck_win = results_df.loc[best_idx, 'pck_window']
    best_pck_glob = results_df.loc[best_idx, 'pck_global']
    
    print(f"Best Block Depth: {best_block} (processed blocks 0-{best_block})")
    print(f"  - PCK Window: {best_pck_win:.2%}")
    print(f"  - PCK Global: {best_pck_glob:.2%}")
    print(f"{'='*60}\n")
    
    print(f"All results saved to: {args.output_dir}")
    print("Block-wise evaluation complete!")


if __name__ == '__main__':
    main()
