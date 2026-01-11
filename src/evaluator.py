import torch
import csv
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import the helpers
from src.loss import predict_keypoints, predict_keypoints_window, denormalize_predictions
from src.pck import compute_pck, get_bbox_size, compute_pck_from_batch

class PCKEvaluator:
    def __init__(self, model, device, dataset=None, cached_dataset=None, window_sizes=[5], temperature=0.02):
        """
        Args:
            model: The model adapter (e.g. DINOv2Adapter, TinySAMAdapter).
            device: 'cuda' or 'cpu'.
            dataset: The dataset to evaluate (optional, required for visualization).
            cached_dataset: Instance of CachedFeaturesDataset (optional).
            window_sizes: List of window sizes for local refinement (default: [5]).
            temperature: Temperature for softmax during inference (default: 0.02).
        """
        self.model = model
        self.device = device
        self.standard_size = self.model.standard_size
        self.dataset = dataset
        self.cached_dataset = cached_dataset
        self.window_sizes = window_sizes if isinstance(window_sizes, list) else [window_sizes]
        self.temperature = temperature

    def _get_features(self, src_name, trg_name):
        """
        Load features for a pair of images.
        Uses cached_dataset if available.
        """
        if self.cached_dataset is None:
            raise ValueError("PCKEvaluator currently requires a cached_dataset for feature extraction.")

        # Load from cache (no augmentation for eval)
        src_data = self.cached_dataset.load_features(src_name, aug_idx=None)
        trg_data = self.cached_dataset.load_features(trg_name, aug_idx=None)

        # Helper to recursively move to device and unsqueeze 2D tensors
        def process_feat(feat):
            if isinstance(feat, dict):
                return {k: process_feat(v) for k, v in feat.items()}
            elif isinstance(feat, (list, tuple)):
                return type(feat)(process_feat(v) for v in feat)
            else:
                f = feat.to(self.device)
                if f.dim() == 2:
                    f = f.unsqueeze(0)
                return f

        src_intermediate = process_feat(src_data['intermediate'])
        trg_intermediate = process_feat(trg_data['intermediate'])

        with torch.no_grad():
            feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
            feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)

        return feat1, feat2, src_data['orig_size'], trg_data['orig_size']

    def compute_metrics_with_sizes(self, feat1, feat2, src_kps_std, batch, trg_sizes, alpha=0.1):
        """
        Pure calculation method used by Trainer.val_step().
        Takes pre-extracted features and computes scalar PCK scores (Global & Window).
        """
        # Ensure inputs are on correct device and dimensions
        src_input = src_kps_std.to(self.device)
        if src_input.dim() == 2: 
            src_input = src_input.unsqueeze(0)

        with torch.no_grad():
            pred_g_norm = predict_keypoints(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                temperature=self.temperature
            )
            pred_w_norm = predict_keypoints_window(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                window_size=self.window_sizes[0],
                temperature=self.temperature
            )

        # Accumulators
        pck_g_accum = 0.0
        pck_w_accum = 0.0
        B = len(trg_sizes)

        for i in range(B):
            w, h = trg_sizes[i]
            
            # Helper to denorm single item: [-1, 1] -> [0, w]
            def denorm(p_norm):
                p = p_norm[i].clone() # (N, 2)
                p[:, 0] = (p[:, 0] + 1) / 2 * (w - 1)
                p[:, 1] = (p[:, 1] + 1) / 2 * (h - 1)
                return p.cpu()

            pred_g_orig = denorm(pred_g_norm)
            pred_w_orig = denorm(pred_w_norm)

            # Use shared PCK logic from src/pck.py
            score_g, _ = compute_pck_from_batch(batch, pred_g_orig, alpha)
            score_w, _ = compute_pck_from_batch(batch, pred_w_orig, alpha)
            
            pck_g_accum += score_g
            pck_w_accum += score_w
        
        return pck_g_accum / B, pck_w_accum / B

    def evaluate(self, dataloader, output_path, alphas, save_top_k=5):
        """
        Runs full evaluation loop (Evaluation Phase).
        Loads features from cache, computes metrics, writes detailed CSV rows,
        and saves visualizations for the best and worst matches.
        """
        self.model.model.eval()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Evaluating on {len(dataloader)} pairs...")
        # Ensure self.dataset is set for visualizations
        self.dataset = dataloader.dataset
        
        print(f"Standard Size for Inference: {self.standard_size}x{self.standard_size}")
        print(f"Saving per-keypoint details to: {output_path}")

        # Open CSV for writing
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Detailed Header
            writer.writerow([
                'pair_id', 'src_img', 'trg_img', 'category', 'kps_idx', 
                'is_visible', 'alpha', 'window_size',
                'is_correct_global', 'is_correct_window'
            ])

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Eval"):
                    self._process_pair(batch, writer, alphas)
        
        print(f"Evaluation finished. CSV Results saved.")
        return output_path

    def _process_pair(self, pair, writer, alphas):
        """
        Internal method for evaluate() loop. Handles data loading, CSV logging,
        and accumulating data for visualization.
        """
        # Unpack Metadata
        pair_id = pair['pair_id']
        src_name = pair['src_name']
        trg_name = pair['trg_name']
        src_kps = pair['src_kps'] 
        trg_kps = pair['trg_kps']
        trg_bbox = pair['trg_bndbox']
        
        # Use local helper for feature loading
        feat1, feat2, src_orig_size, trg_orig_size = self._get_features(src_name, trg_name)
        
        trg_orig_w, trg_orig_h = trg_orig_size
        src_orig_w, src_orig_h = src_orig_size
        
        # Scale Source Keypoints: Original -> Standard
        src_kps_std = src_kps.clone()
        src_kps_std[:, 0] *= (self.standard_size / src_orig_w)
        src_kps_std[:, 1] *= (self.standard_size / src_orig_h)
        src_input = src_kps_std.to(self.device).unsqueeze(0)
        
        # Predict Global Method
        pred_global_norm = predict_keypoints(
            feat1, feat2, src_input,
            src_img_size=(self.standard_size, self.standard_size),
            temperature=self.temperature
        )
        
        # Denormalize global prediction
        pred_global_orig = denormalize_predictions(pred_global_norm, trg_orig_size).squeeze(0).cpu()
        
        # Compute visibility mask
        visibility_mask = (trg_kps[:, 0] > 0) & (trg_kps[:, 1] > 0)
        category = pair.get('category', 'unknown')

        # Loop over window sizes
        for window_size in self.window_sizes:
            # Predict Window Method (computed for each window size)
            pred_window_norm = predict_keypoints_window(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                window_size=window_size,
                temperature=self.temperature
            )
            
            # Denormalize window prediction
            pred_window_orig = denormalize_predictions(pred_window_norm, trg_orig_size).squeeze(0).cpu()
            
            # Loop over alpha values
            for alpha in alphas:
                # Compute PCK scores using standardized functions
                _, correct_mask_global = compute_pck(
                    pred_global_orig,
                    trg_kps,
                    trg_bbox,
                    alpha,
                    visibility_mask
                )
                
                _, correct_mask_window = compute_pck(
                    pred_window_orig,
                    trg_kps,
                    trg_bbox,
                    alpha,
                    visibility_mask
                )
                
                # Logging to CSV
                num_kps = correct_mask_global.shape[0]
                for k in range(num_kps):
                    is_visible = visibility_mask[k].item()
                    is_correct_g = correct_mask_global[k].item()
                    is_correct_w = correct_mask_window[k].item()
                    
                    if is_visible:
                        writer.writerow([
                            pair_id, src_name, trg_name, category, k, int(is_visible), alpha, window_size,
                            int(is_correct_g), int(is_correct_w)
                        ])

    def process_results(csv_path, save_dir):
        """
        Reads the generated CSV and prints summary metrics for both methods.
        Groups results by alpha value if alpha column exists.
        """
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            print("CSV is empty.")
            return

        if 'is_visible' in df.columns:
            n_total = len(df)
            df = df[df['is_visible'] == 1]
            print(f"Filtered invisible keypoints: {n_total} -> {len(df)}")

        # Get unique alpha and window_size values
        unique_alphas = sorted(df['alpha'].unique())
        unique_window_sizes = sorted(df['window_size'].unique())
        
        print(f"\nFound {len(unique_alphas)} alpha value(s): {unique_alphas}")
        print(f"Found {len(unique_window_sizes)} window size(s): {unique_window_sizes}")
        
        # Iterate over all combinations of alpha and window_size
        for alpha_val in unique_alphas:
            for window_size_val in unique_window_sizes:
                # Filter dataframe by current alpha and window_size
                df_subset = df[(df['alpha'] == alpha_val) & (df['window_size'] == window_size_val)]
                
                # Print header for this combination
                print(f"\n{'='*105}")
                print(f"RESULTS FOR ALPHA={alpha_val}, WINDOW_SIZE={window_size_val}")
                print(f"{'='*105}")
                alpha_suffix = f"_alpha_{alpha_val}_ws_{window_size_val}"

                def print_summary(df_subset):
                    # 1. Global Metrics
                    pck_kps_g = df_subset['is_correct_global'].mean()
                    img_scores_g = df_subset.groupby(['pair_id', 'src_img'])['is_correct_global'].mean()
                    pck_img_g = img_scores_g.mean()
                    
                    # 2. Window Metrics
                    pck_kps_w = df_subset['is_correct_window'].mean()
                    img_scores_w = df_subset.groupby(['pair_id', 'src_img'])['is_correct_window'].mean()
                    pck_img_w = img_scores_w.mean()
                    
                    print(f"\nTotal Keypoints Evaluated: {len(df_subset)}")
                    print(f"Total Images Evaluated:    {len(img_scores_g)}")
                    print("-" * 60)
                    print(f"{'METRIC':<20} | {'GLOBAL':<10} | {'WINDOW':<10} | {'DELTA':<10}")
                    print("-" * 60)
                    print(f"{'PCK (Per Keypoint)':<20} | {pck_kps_g:.2%}     | {pck_kps_w:.2%}     | {pck_kps_w - pck_kps_g:+.2%}")
                    print(f"{'PCK (Per Image)':<20} | {pck_img_g:.2%}     | {pck_img_w:.2%}     | {pck_img_w - pck_img_g:+.2%}")
                    print("-" * 60)

                def print_by_category(df_subset, alpha_suffix=""):
                    cat_pck_kps_g = df_subset.groupby('category')['is_correct_global'].mean()
                    cat_pck_kps_w = df_subset.groupby('category')['is_correct_window'].mean()
                    
                    # Formula: Average of (Correct / Visible) for each image pair
                    img_scores = df_subset.groupby(['pair_id', 'category'])[['is_correct_global', 'is_correct_window']].mean().reset_index()
                    
                    cat_pck_img_g = img_scores.groupby('category')['is_correct_global'].mean()
                    cat_pck_img_w = img_scores.groupby('category')['is_correct_window'].mean()

                    # Assemble Final Table
                    summary = pd.DataFrame({
                        'Img_Global': cat_pck_img_g,
                        'Img_Window': cat_pck_img_w,
                        'Kps_Global': cat_pck_kps_g,
                        'Kps_Window': cat_pck_kps_w,
                        'Num_Images': img_scores['category'].value_counts()
                    })
                    
                    # Sort by PCK Image Window score
                    summary = summary.sort_values('Img_Window', ascending=False)

                    print("\n" + "="*105)
                    print(f"{'CATEGORY':<20} | {'IMG (G)':<10} | {'IMG (W)':<10} | {'KPS (G)':<10} | {'KPS (W)':<10} | {'# IMGS':<8}")
                    print("-" * 105)
                    
                    for cat, row in summary.iterrows():
                        print(f"{cat:<20} | {row['Img_Global']:<10.2%} | {row['Img_Window']:<10.2%} | {row['Kps_Global']:<10.2%} | {row['Kps_Window']:<10.2%} | {int(row['Num_Images']):<8}")
                        
                    print("-" * 105)
                    # Use columns directly from dataframes to ensure overall mean is accurate across all samples
                    global_img_g = img_scores['is_correct_global'].mean()
                    global_img_w = img_scores['is_correct_window'].mean()
                    global_kps_g = df_subset['is_correct_global'].mean()
                    global_kps_w = df_subset['is_correct_window'].mean()
                    
                    print(f"{'OVERALL (Mean)':<20} | {global_img_g:<10.2%} | {global_img_w:<10.2%} | {global_kps_g:<10.2%} | {global_kps_w:<10.2%} | {len(img_scores)}")
                    print("=" * 105)

                    # Save to file
                    by_category_output_path = os.path.join(save_dir, f'summary_by_category{alpha_suffix}.csv')
                    summary.to_csv(by_category_output_path)
                    print(f"\nBy category table saved to: {by_category_output_path}")

                # Print results for this combination
                print_summary(df_subset)
                print_by_category(df_subset, alpha_suffix)

    def _plot_pair(self, res, save_path):
        """
        Helper to draw Source (Query) and Target (Prediction vs GT).
        """
        
        # Load Images
        src_img_path = str(self.dataset.get_image_path(res['src_path']))
        trg_img_path = str(self.dataset.get_image_path(res['trg_path']))
        
        src_img = cv2.imread(src_img_path)
        trg_img = cv2.imread(trg_img_path)

        if src_img is None or trg_img is None:
            print(f"Could not load image for visualization: {src_img_path} or {trg_img_path}")
            return

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        trg_img = cv2.cvtColor(trg_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # 1. Source Image + Query Points
        ax[0].imshow(src_img)
        ax[0].set_title("Source (Query Keypoints)")
        ax[0].axis('off')
        
        # Draw source bounding box if available
        if 'src_bbox' in res:
            src_bbox = res['src_bbox']
            rect = Rectangle((src_bbox[0], src_bbox[1]), 
                            src_bbox[2] - src_bbox[0], 
                            src_bbox[3] - src_bbox[1],
                            linewidth=2, edgecolor='cyan', facecolor='none', label='BBox')
            ax[0].add_patch(rect)
        
        # Plot source points (Yellow)
        # Filter invisible source points (often 0,0)
        src_kps = res['src_kps']
        valid_src = (src_kps[:, 0] > 0) & (src_kps[:, 1] > 0)
        ax[0].scatter(src_kps[valid_src, 0], src_kps[valid_src, 1], 
                      c='yellow', s=50, marker='o', edgecolors='black', label='Query')
        ax[0].legend(loc='lower right', fontsize='small')

        # 2. Target Image + Pred vs GT
        ax[1].imshow(trg_img)
        ax[1].set_title(f"Target (PCK: {res['pck']:.2f})")
        ax[1].axis('off')
        
        # Draw target bounding box if available
        if 'trg_bbox' in res:
            trg_bbox = res['trg_bbox']
            rect = Rectangle((trg_bbox[0], trg_bbox[1]), 
                            trg_bbox[2] - trg_bbox[0], 
                            trg_bbox[3] - trg_bbox[1],
                            linewidth=2, edgecolor='cyan', facecolor='none', label='BBox')
            ax[1].add_patch(rect)

        # Plot Ground Truth (Green)
        gt = res['trg_kps']
        valid_gt = (gt[:, 0] > 0) & (gt[:, 1] > 0)
        ax[1].scatter(gt[valid_gt, 0], gt[valid_gt, 1], 
                      c='lime', s=60, marker='o', edgecolors='black', label='Ground Truth')
        
        # Plot Prediction (Red)
        pred = res['pred_kps']
        # Only plot predictions where GT was valid (to reduce clutter)
        ax[1].scatter(pred[valid_gt, 0], pred[valid_gt, 1], 
                      c='red', s=40, marker='x', linewidth=2, label='Prediction')

        # Draw lines connecting GT to Pred
        for j in range(len(gt)):
            if valid_gt[j]:
                ax[1].plot([gt[j, 0], pred[j, 0]], [gt[j, 1], pred[j, 1]], 
                           c='white', alpha=0.4, linestyle='--', linewidth=1)

        ax[1].legend(loc='lower right', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig)

    def evaluate_pair_by_id(self, pair_id, output_dir="single_evals", alpha=0.1):
        """
        Runs evaluation on a single pair by its pair_id.
        Computes AND Visualizes both Global and Window methods.
        
        Args:
            pair_id: The pair_id string to evaluate (e.g., 'trn:aeroplane-0001:aeroplane-0010')
            output_dir: Directory to save visualization outputs
            alpha: Transparency for heatmap overlay
        """
        try:
            sample = self.dataset.get_pair_by_id(pair_id)
        except KeyError as e:
            print(f"Error: {e}")
            return

        s_name = sample['src_name']
        t_name = sample['trg_name']
        
        # Load Raw Data
        src_kps_raw = torch.tensor(sample['src_kps'], dtype=torch.float32)
        trg_kps_raw = torch.tensor(sample['trg_kps'], dtype=torch.float32)
        trg_bbox = torch.tensor(sample['trg_bndbox'], dtype=torch.float32)

        print(f"\n--- Evaluating Pair ID: {pair_id} ---")
        print(f"Source: {s_name}")
        print(f"Target: {t_name}")

        # Load Features
        feat1, feat2, src_orig_size, trg_orig_size = self._get_features(s_name, t_name)

        # Prepare keypoints (no visibility filtering for eval)
        src_kps_normed = src_kps_raw.clone()
        trg_kps_normed = trg_kps_raw.clone()

        trg_orig_w, trg_orig_h = trg_orig_size
        src_orig_w, src_orig_h = src_orig_size
        
        # Scale Source Keypoints: Original -> Standard
        src_kps_std = src_kps_raw.clone()
        src_kps_std[:, 0] *= (self.standard_size / src_orig_w)
        src_kps_std[:, 1] *= (self.standard_size / src_orig_h)
        src_input = src_kps_std.to(self.device).unsqueeze(0)

        # Global method
        pred_global_norm = predict_keypoints(
            feat1, feat2, src_input,
            src_img_size=(self.standard_size, self.standard_size),
            temperature=self.temperature
        )
        pred_kps_global = denormalize_predictions(pred_global_norm, trg_orig_size).squeeze(0).cpu()

        # Window method
        pred_window_norm = predict_keypoints_window(
            feat1, feat2, src_input,
            src_img_size=(self.standard_size, self.standard_size),
            window_size=self.window_sizes[0],
            temperature=self.temperature
        )
        pred_kps_window = denormalize_predictions(pred_window_norm, trg_orig_size).squeeze(0).cpu()

        # Compute PCK
        pck_global, _ = compute_pck(
            pred_kps_global, trg_kps_normed, trg_bbox, alpha=alpha
        )
        pck_window, _ = compute_pck(
            pred_kps_window, trg_kps_normed, trg_bbox, alpha=alpha
        )

        print(f"PCK (Global): {pck_global:.4f}")
        print(f"PCK (Window): {pck_window:.4f}")
        
        # Calculate Errors (Mean pixel distance for valid points)
        # Using simple L2 distance for matched points
        valid_mask = (trg_kps_normed[:, 0] > 0) & (trg_kps_normed[:, 1] > 0)
        
        if valid_mask.sum() > 0:
            err_g = (pred_kps_global[valid_mask] - trg_kps_normed[valid_mask]).norm(dim=1).mean().item()
            err_w = (pred_kps_window[valid_mask] - trg_kps_normed[valid_mask]).norm(dim=1).mean().item()
        else:
            err_g = 0.0
            err_w = 0.0

        # Visualize
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{pair_id.replace(':', '_')}_comparison.png")

        # Reuse visualization dictionary format expected by _plot_pair_comparison
        res = {
            'src_path': s_name,
            'trg_path': t_name,
            'src_kps': src_kps_raw.numpy(),
            'trg_kps': trg_kps_raw.numpy(),
            'src_bbox': sample.get('src_bndbox'),
            'trg_bbox': sample['trg_bndbox'], # gt bbox
            'pred_global': pred_kps_global.numpy(),
            'pred_window': pred_kps_window.numpy(),
            'pck_g': pck_global,
            'pck_w': pck_window,
            'err_g': err_g,
            'err_w': err_w
        }

        self._plot_pair_comparison(res, out_path)
        print(f"Saved comparison to: {out_path}")

    def evaluate_pair_by_index(self, pair_idx, output_dir="single_evals", alpha=0.1):
        """
        Runs evaluation on a single pair by index.
        Computes AND Visualizes both Global and Window methods.
        """
        # Retrieve the single sample
        if pair_idx < 0 or pair_idx >= len(self.dataset):
            print(f"Error: Index {pair_idx} out of bounds.")
            return

        # Get pair_id from index and delegate to evaluate_pair_by_id
        pair_id = self.dataset.pair_ids[pair_idx]
        print(f"Pair index {pair_idx} corresponds to pair_id: {pair_id}")
        self.evaluate_pair_by_id(pair_id, output_dir=output_dir, alpha=alpha)

    def _plot_pair_comparison(self, res, save_path):
        """
        Generates a 3-column plot: [Source] | [Global Pred] | [Window Pred]
        """
        
        src_img_path = str(self.dataset.get_image_path(res["src_path"]))
        trg_img_path = str(self.dataset.get_image_path(res["trg_path"]))

        src_img = cv2.imread(src_img_path)
        trg_img = cv2.imread(trg_img_path)

        if src_img is None or trg_img is None:
            return

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        trg_img = cv2.cvtColor(trg_img, cv2.COLOR_BGR2RGB)
        
        # Create 3 subplots
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # --- Source ---
        ax[0].imshow(src_img)
        ax[0].set_title("Source (Query)")
        ax[0].axis('off')
        
        # Draw source bounding box if available
        if 'src_bbox' in res:
            src_bbox = res['src_bbox']
            rect = Rectangle((src_bbox[0], src_bbox[1]), 
                            src_bbox[2] - src_bbox[0], 
                            src_bbox[3] - src_bbox[1],
                            linewidth=2, edgecolor='cyan', facecolor='none')
            ax[0].add_patch(rect)
        
        src_kps = res['src_kps']
        valid_src = (src_kps[:, 0] > 0) & (src_kps[:, 1] > 0)
        ax[0].scatter(src_kps[valid_src, 0], src_kps[valid_src, 1], c='yellow', s=50, edgecolors='black')

        # Shared Helper for Target plotting
        def plot_target(ax_idx, title, pred_kps, pck, err):
            ax[ax_idx].imshow(trg_img)
            ax[ax_idx].set_title(f"{title}\nPCK: {pck:.2f}, Err: {err:.1f}px")
            ax[ax_idx].axis('off')
            
            # Draw target bounding box if available
            if 'trg_bbox' in res:
                trg_bbox = res['trg_bbox']
                rect = Rectangle((trg_bbox[0], trg_bbox[1]), 
                                trg_bbox[2] - trg_bbox[0], 
                                trg_bbox[3] - trg_bbox[1],
                                linewidth=2, edgecolor='cyan', facecolor='none')
                ax[ax_idx].add_patch(rect)
            
            gt = res['trg_kps']
            valid_gt = (gt[:, 0] > 0) & (gt[:, 1] > 0)
            
            # Ground Truth (Green)
            ax[ax_idx].scatter(gt[valid_gt, 0], gt[valid_gt, 1], c='lime', s=60, edgecolors='black', label='GT')
            
            # Prediction (Red)
            ax[ax_idx].scatter(pred_kps[valid_gt, 0], pred_kps[valid_gt, 1], c='red', s=40, marker='x', label='Pred')
            
            # Connectors
            for j in range(len(gt)):
                if valid_gt[j]:
                    ax[ax_idx].plot([gt[j, 0], pred_kps[j, 0]], [gt[j, 1], pred_kps[j, 1]], c='white', alpha=0.4, linestyle='--')
            
            ax[ax_idx].legend(loc='lower right', fontsize='small')

        # --- Global Result ---
        plot_target(1, "Global Method", res['pred_global'], res['pck_g'], res['err_g'])

        # --- Window Result ---
        plot_target(2, "Window Method", res['pred_window'], res['pck_w'], res['err_w'])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)