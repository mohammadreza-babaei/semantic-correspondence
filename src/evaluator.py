import torch
import csv
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import the helpers
from src.new_loss import predict_keypoints, predict_keypoints_window
from src.pck import compute_pck_from_batch

class PCKEvaluator:
    def __init__(self, trainer, device):
        """
        Args:
            trainer: The Trainer instance (holds .features_cache and .model).
            device: 'cuda' or 'cpu'.
        """
        self.trainer = trainer
        self.model = trainer.model # The adapter
        self.device = device
        # Use the model's configured standard size (e.g., 518 or 528)
        self.standard_size = getattr(self.model, 'standard_size', 518)

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

    def evaluate(self, dataloader, output_path, alpha=0.1):
        """
        Runs full evaluation loop (Evaluation Phase).
        Loads features from cache, computes metrics, and writes detailed CSV rows.
        """
        self.model.model.eval()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Evaluating on {len(dataloader)} pairs...")
        print(f"Standard Size for Inference: {self.standard_size}x{self.standard_size}")
        print(f"Saving per-keypoint details to: {output_path}")

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
        
        print(f"Evaluation finished. Results saved.")
        return output_path

    def _process_batch(self, batch, batch_idx, writer, alpha):
        """
        Internal method for evaluate() loop. Handles data loading and CSV logging.
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
            
            # Use Disk Loader
            try:
                cache_src = self.trainer._load_cached_features(s_name)
                cache_trg = self.trainer._load_cached_features(t_name)
            except RuntimeError:
                print(f"Warning: Cache file missing for {s_name} or {t_name}. Skipping.")
                continue
            
            src_orig_w, src_orig_h = cache_src['orig_size']
            trg_orig_w, trg_orig_h = cache_trg['orig_size']
            
            # Load Features & Forward Pass
            src_inter = cache_src['intermediate'].to(self.device)
            trg_inter = cache_trg['intermediate'].to(self.device)
            
            # --- FIX: Only unsqueeze if it is missing a batch dim entirely (Dim=2) ---
            # Your cache already saves with batch dim 1, so src_inter is typically (1, N, C)
            if src_inter.dim() == 2: 
                src_inter = src_inter.unsqueeze(0)
            if trg_inter.dim() == 2: 
                trg_inter = trg_inter.unsqueeze(0)
            # ------------------------------------------------------------------------

            feat1 = self.model.forward_unfrozen_blocks(src_inter)
            feat2 = self.model.forward_unfrozen_blocks(trg_inter)
            
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
                temperature=0.1
            )
            
            # Scale Prediction: [-1, 1] -> Target Original
            def denorm(p_norm):
                p = p_norm.clone()
                p[..., 0] = (p_norm[..., 0] + 1) / 2 * (trg_orig_w - 1)
                p[..., 1] = (p_norm[..., 1] + 1) / 2 * (trg_orig_h - 1)
                return p.squeeze(0).cpu()

            pred_global_orig = denorm(pred_global_norm)
            pred_window_orig = denorm(pred_window_norm)
            
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
            
            # Log to CSV
            category = batch['category'][i] if (B > 1 and 'category' in batch) else batch.get('category', 'unknown')
            if isinstance(category, list): category = category[0]
            
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

    def summarize_results(self, csv_path, alpha):
        """
        Reads the generated CSV and prints summary metrics for both methods.
        """
        if not os.path.exists(csv_path):
            print("CSV not found.")
            return

        print("\n" + "-"*60)
        print(f"COMPUTING SUMMARY METRICS (Alpha={alpha})")
        print("-" * 60)
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
            
        if len(df) == 0:
            print("CSV is empty.")
            return

        if 'is_visible' in df.columns:
            df = df[df['is_visible'] == 1]
        
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