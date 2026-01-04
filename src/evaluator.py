import torch
import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import the helpers
from src.new_loss import predict_keypoints, predict_keypoints_window, denormalize_predictions
from src.pck import compute_pck_from_batch

class PCKEvaluator:
    def __init__(self, trainer, device, dataset=None, window_size=5):
        """
        Args:
            trainer: The Trainer instance (holds .features_cache and .model).
            device: 'cuda' or 'cpu'.
            dataset: The dataset to evaluate (optional).
            window_size: Size of the window for local refinement in predict_keypoints_window (default: 5).
        """
        self.trainer = trainer
        self.model = trainer.model # The adapter
        self.device = device
        # Use the model's configured standard size (e.g., 518 or 528)
        self.standard_size = getattr(self.model, 'standard_size', 518)
        self.dataset = dataset
        self.window_size = window_size

        # Buffer to store results for visualization
        self.pair_results = []

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
            # 1. Global Prediction
            pred_g_norm = predict_keypoints(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                temperature=0.1
            )
            # 2. Window Prediction
            pred_w_norm = predict_keypoints_window(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                window_size=self.window_size,
                temperature=0.1
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

    def evaluate(self, dataloader, output_path, alpha=0.1, save_top_k=5):
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

        # Reset results buffer
        self.pair_results = []

        # Open CSV for writing
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Detailed Header
            writer.writerow([
                'pair_idx', 'src_img', 'trg_img', 'category', 'kps_idx', 
                'bbox_size', 'threshold', 'is_visible',
                'pixel_error_global', 'is_correct_global', 
                'pixel_error_window', 'is_correct_window'
            ])

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval")):
                    self._process_batch(batch, batch_idx, writer, alpha)
        
        print(f"Evaluation finished. CSV Results saved.")
        
        # --- Visualization of Extremes ---
        if save_top_k > 0:
            vis_dir = os.path.join(os.path.dirname(output_path), "visualizations")
            self.save_extremes(vis_dir, k=save_top_k)

        return output_path

    def _process_batch(self, batch, batch_idx, writer, alpha):
        """
        Internal method for evaluate() loop. Handles data loading, CSV logging,
        and accumulating data for visualization.
        """
        # 1. Unpack Metadata
        src_names = batch['src_name']
        trg_names = batch['trg_name']
        src_kps_raw = batch['src_kps']   # (B, N, 2)
        trg_kps_raw = batch['trg_kps']   # (B, N, 2)
        trg_bbox = batch['trg_bndbox']   # (B, 4)
        
        B = len(src_names) if isinstance(src_names, list) else 1
        
        for i in range(B):
            s_name = src_names[i] if B > 1 else src_names
            t_name = trg_names[i] if B > 1 else trg_names
            
            # Use trainer's helper for feature loading (with error handling)
            try:
                feat1, feat2, src_orig_size, trg_orig_size = self.trainer._prepare_feature_pair(
                    s_name, t_name, requires_grad=False
                )
            except RuntimeError:
                print(f"Warning: Cache file missing for {s_name} or {t_name}. Skipping.")
                continue
            
            trg_orig_w, trg_orig_h = trg_orig_size
            src_orig_w, src_orig_h = src_orig_size
            
            # Scale Source Keypoints: Original -> Standard
            curr_src_kps = src_kps_raw[i] if B > 1 else src_kps_raw
            if not isinstance(curr_src_kps, torch.Tensor): 
                curr_src_kps = torch.tensor(curr_src_kps, dtype=torch.float32)
                
            src_kps_std = curr_src_kps.clone()
            src_kps_std[:, 0] *= (self.standard_size / src_orig_w)
            src_kps_std[:, 1] *= (self.standard_size / src_orig_h)
            src_input = src_kps_std.to(self.device).unsqueeze(0)
            
            # Predict (Both Methods)
            pred_global_norm = predict_keypoints(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                temperature=0.1
            )
            pred_window_norm = predict_keypoints_window(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                window_size=self.window_size,
                temperature=0.1
            )
            
            # Denormalize predictions using shared helper
            pred_global_orig = denormalize_predictions(pred_global_norm, trg_orig_size).squeeze(0).cpu()
            pred_window_orig = denormalize_predictions(pred_window_norm, trg_orig_size).squeeze(0).cpu()
            
            # Compute Errors
            curr_trg_kps = trg_kps_raw[i] if B > 1 else trg_kps_raw
            if not isinstance(curr_trg_kps, torch.Tensor):
                curr_trg_kps = torch.tensor(curr_trg_kps)
                
            dist_global = torch.norm(pred_global_orig - curr_trg_kps, dim=-1)
            dist_window = torch.norm(pred_window_orig - curr_trg_kps, dim=-1)
            
            # Compute Threshold
            curr_bbox = trg_bbox[i] if B > 1 else trg_bbox
            if not isinstance(curr_bbox, torch.Tensor): curr_bbox = torch.tensor(curr_bbox)
            bbox_size = max(curr_bbox[2]-curr_bbox[0], curr_bbox[3]-curr_bbox[1]).item()
            threshold = alpha * bbox_size
            
            # Logging to CSV & Accumulating Pair Stats
            category = batch['category'][i] if (B > 1 and 'category' in batch) else batch.get('category', 'unknown')
            if isinstance(category, list): category = category[0]
            
            # Stats accumulators for visualization ranking
            pair_correct_count = 0
            pair_visible_count = 0
            pair_total_error = 0.0

            num_kps = dist_global.shape[0]
            for k in range(num_kps):
                err_g = dist_global[k].item()
                err_w = dist_window[k].item()
                is_visible = (curr_trg_kps[k, 0] > 0) and (curr_trg_kps[k, 1] > 0)
                is_correct_g = (err_g <= threshold) and is_visible
                is_correct_w = (err_w <= threshold) and is_visible
                
                if is_visible:
                    writer.writerow([
                        batch_idx, s_name, t_name, category, k, 
                        f"{bbox_size:.2f}", f"{threshold:.4f}", int(is_visible),
                        f"{err_g:.4f}", int(is_correct_g),
                        f"{err_w:.4f}", int(is_correct_w)
                    ])
                    
                    # For visualization ranking, we primarily use the Window method score
                    pair_visible_count += 1
                    pair_total_error += err_w
                    if is_correct_w:
                        pair_correct_count += 1

            # --- Store Pair Results for Best/Worst Analysis ---
            if pair_visible_count > 0:
                pck_score = pair_correct_count / pair_visible_count
                avg_error = pair_total_error / pair_visible_count
                
                self.pair_results.append({
                    'src_path': s_name,
                    'trg_path': t_name,
                    'src_kps': curr_src_kps.numpy(),
                    'trg_kps': curr_trg_kps.numpy(),
                    'pred_kps': pred_window_orig.numpy(), # Using Window prediction for vis
                    'pck': pck_score,
                    'error': avg_error
                })

    def process_results(csv_path, save_dir):
        """
        Reads the generated CSV and prints summary metrics for both methods.
        """
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            print("CSV is empty.")
            return

        if 'is_visible' in df.columns:
            n_total = len(df)
            df = df[df['is_visible'] == 1]
            print(f"Filtered invisible keypoints: {n_total} -> {len(df)}")

        def print_summary():
            # 1. Global Metrics
            pck_kps_g = df['is_correct_global'].mean()
            img_scores_g = df.groupby(['pair_idx', 'src_img'])['is_correct_global'].mean()
            pck_img_g = img_scores_g.mean()
            
            # 2. Window Metrics
            pck_kps_w = df['is_correct_window'].mean()
            img_scores_w = df.groupby(['pair_idx', 'src_img'])['is_correct_window'].mean()
            pck_img_w = img_scores_w.mean()
            
            print(f"Total Keypoints Evaluated: {len(df)}")
            print(f"Total Images Evaluated:    {len(img_scores_g)}")
            print("-" * 60)
            print(f"{'METRIC':<20} | {'GLOBAL':<10} | {'WINDOW':<10} | {'DELTA':<10}")
            print("-" * 60)
            print(f"{'PCK (Per Keypoint)':<20} | {pck_kps_g:.2%}     | {pck_kps_w:.2%}     | {pck_kps_w - pck_kps_g:+.2%}")
            print(f"{'PCK (Per Image)':<20} | {pck_img_g:.2%}     | {pck_img_w:.2%}     | {pck_img_w - pck_img_g:+.2%}")
            print("-" * 60)

        def print_by_category():
            # Metric 1: Per-Keypoint PCK (Global, Window & Per Category)
            cat_pck_kps_g = df.groupby('category')['is_correct_global'].mean()
            cat_pck_kps_w = df.groupby('category')['is_correct_window'].mean()
            
            # Metric 2: Per-Image PCK (Global, Window & Per Category)
            # Formula: Average of (Correct / Visible) for each image pair
            img_scores = df.groupby(['pair_idx', 'category'])[['is_correct_global', 'is_correct_window']].mean().reset_index()
            
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

            print("="*105)
            print(f"{'CATEGORY':<20} | {'IMG (G)':<10} | {'IMG (W)':<10} | {'KPS (G)':<10} | {'KPS (W)':<10} | {'# IMGS':<8}")
            print("-" * 105)
            
            for cat, row in summary.iterrows():
                print(f"{cat:<20} | {row['Img_Global']:<10.2%} | {row['Img_Window']:<10.2%} | {row['Kps_Global']:<10.2%} | {row['Kps_Window']:<10.2%} | {int(row['Num_Images']):<8}")
                
            print("-" * 105)
            # Use columns directly from dataframes to ensure overall mean is accurate across all samples
            global_img_g = img_scores['is_correct_global'].mean()
            global_img_w = img_scores['is_correct_window'].mean()
            global_kps_g = df['is_correct_global'].mean()
            global_kps_w = df['is_correct_window'].mean()
            
            print(f"{'OVERALL (Mean)':<20} | {global_img_g:<10.2%} | {global_img_w:<10.2%} | {global_kps_g:<10.2%} | {global_kps_w:<10.2%} | {len(img_scores)}")
            print("=" * 105)

            # Save to file
            by_category_output_path = os.path.join(save_dir, 'summary_by_category.csv')
            summary.to_csv(by_category_output_path)
            print(f"\nBy category table saved to: {by_category_output_path}")

        print_summary()
        print_by_category()

    def save_extremes(self, save_dir, k=5):
        """
        Sorts tracked results and visualizes Top K and Bottom K image pairs.
        """
        if not self.pair_results:
            print("No results available to visualize.")
            return

        print(f"Generating visualizations for Top {k} and Bottom {k} results...")
        os.makedirs(os.path.join(save_dir, 'best'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'worst'), exist_ok=True)

        # Sort: Primary key = PCK (Desc), Secondary key = Error (Asc -> using negative for sort)
        sorted_res = sorted(self.pair_results, key=lambda x: (x['pck'], -x['error']), reverse=True)

        best_pairs = sorted_res[:k]
        worst_pairs = sorted_res[-k:]

        for i, res in enumerate(best_pairs):
            filename = f"rank{i+1}_best_pck{res['pck']:.2f}.png"
            self._plot_pair(res, os.path.join(save_dir, 'best', filename))

        for i, res in enumerate(worst_pairs):
            # Reverse rank index for worst (rank 1 = absolute worst)
            rank = len(sorted_res) - (len(worst_pairs) - 1 - i)
            filename = f"rank{rank}_worst_pck{res['pck']:.2f}.png"
            self._plot_pair(res, os.path.join(save_dir, 'worst', filename))
        
        print(f"Visualizations saved to {save_dir}")

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
        
        # Plot source points (Yellow)
        # Filter invisible source points (often 0,0)
        src_kps = res['src_kps']
        valid_src = (src_kps[:, 0] > 0) & (src_kps[:, 1] > 0)
        ax[0].scatter(src_kps[valid_src, 0], src_kps[valid_src, 1], 
                      c='yellow', s=50, marker='o', edgecolors='black', label='Query')

        # 2. Target Image + Pred vs GT
        ax[1].imshow(trg_img)
        ax[1].set_title(f"Target (PCK: {res['pck']:.2f}, Err: {res['error']:.1f}px)")
        ax[1].axis('off')

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

    def evaluate_pair_by_index(self, pair_idx, output_dir="single_evals", alpha=0.1):
        """
        Runs evaluation on a single pair by index.
        Computes AND Visualizes both Global and Window methods.
        """
        # 1. Retrieve the single sample
        if pair_idx < 0 or pair_idx >= len(self.dataset):
            print(f"Error: Index {pair_idx} out of bounds.")
            return

        sample = self.dataset[pair_idx]
        s_name = sample['src_name']
        t_name = sample['trg_name']
        
        # Load Raw Data
        src_kps_raw = torch.tensor(sample['src_kps'], dtype=torch.float32)
        trg_kps_raw = torch.tensor(sample['trg_kps'], dtype=torch.float32)
        trg_bbox = torch.tensor(sample['trg_bndbox'], dtype=torch.float32)

        print(f"\n--- Evaluating Pair Index: {pair_idx} ---")
        print(f"Source: {s_name}")
        print(f"Target: {t_name}")

        # 2. Load Features
        feat1, feat2, src_orig_size, trg_orig_size = self.trainer._prepare_feature_pair(
            s_name, t_name, requires_grad=False
        )

        # 3. Prepare keypoints (no visibility filtering for eval)
        src_kps_std, _ = self.trainer._prepare_keypoints(
            src_kps_raw, trg_kps_raw, src_orig_size, trg_orig_size, filter_visibility=False
        )
        src_input = src_kps_std.unsqueeze(0)  # Add batch dimension

        # 4. Model Inference (BOTH Methods)
        self.model.model.eval()
        with torch.no_grad():
            # A. Global Prediction
            pred_global_norm = predict_keypoints(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                temperature=0.1
            )
            
            # B. Window Prediction
            pred_window_norm = predict_keypoints_window(
                feat1, feat2, src_input,
                src_img_size=(self.standard_size, self.standard_size),
                window_size=self.window_size,
                temperature=0.1
            )

        # 5. Denormalize using shared helper
        pred_g_orig = denormalize_predictions(pred_global_norm, trg_orig_size).squeeze(0).cpu()
        pred_w_orig = denormalize_predictions(pred_window_norm, trg_orig_size).squeeze(0).cpu()
        
        # 6. Compute Metrics for Both
        def calc_score(pred_kps):
            dist = torch.norm(pred_kps - trg_kps_raw, dim=-1)
            bbox_size = max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1]).item()
            threshold = alpha * bbox_size
            valid_mask = (trg_kps_raw[:, 0] > 0) & (trg_kps_raw[:, 1] > 0)
            valid_dist = dist[valid_mask]
            
            if len(valid_dist) > 0:
                pck = (valid_dist <= threshold).float().mean().item()
                err = valid_dist.mean().item()
            else:
                pck, err = 0.0, 0.0
            return pck, err

        pck_g, err_g = calc_score(pred_g_orig)
        pck_w, err_w = calc_score(pred_w_orig)

        print("-" * 40)
        print(f"{'Method':<10} | {'PCK':<10} | {'Avg Error':<10}")
        print("-" * 40)
        print(f"{'Global':<10} | {pck_g:.2%}     | {err_g:.2f} px")
        print(f"{'Window':<10} | {pck_w:.2%}     | {err_w:.2f} px")
        print("-" * 40)

        # 6. Visualize Comparison
        res = {
            'src_path': s_name,
            'trg_path': t_name,
            'src_kps': src_kps_raw.numpy(),
            'trg_kps': trg_kps_raw.numpy(),
            'pred_global': pred_g_orig.numpy(),
            'pred_window': pred_w_orig.numpy(),
            'pck_g': pck_g, 'err_g': err_g,
            'pck_w': pck_w, 'err_w': err_w
        }
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"pair_{pair_idx}_comparison.png")
        self._plot_pair_comparison(res, save_path)
        print(f"Comparison image saved to: {save_path}")

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

        # --- 1. Source ---
        ax[0].imshow(src_img)
        ax[0].set_title("Source (Query)")
        ax[0].axis('off')
        src_kps = res['src_kps']
        valid_src = (src_kps[:, 0] > 0) & (src_kps[:, 1] > 0)
        ax[0].scatter(src_kps[valid_src, 0], src_kps[valid_src, 1], c='yellow', s=50, edgecolors='black')

        # Shared Helper for Target plotting
        def plot_target(ax_idx, title, pred_kps, pck, err):
            ax[ax_idx].imshow(trg_img)
            ax[ax_idx].set_title(f"{title}\nPCK: {pck:.2f}, Err: {err:.1f}px")
            ax[ax_idx].axis('off')
            
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

        # --- 2. Global Result ---
        plot_target(1, "Global Method", res['pred_global'], res['pck_g'], res['err_g'])

        # --- 3. Window Result ---
        plot_target(2, "Window Method", res['pred_window'], res['pck_w'], res['err_w'])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)