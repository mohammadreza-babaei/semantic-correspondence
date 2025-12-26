import torch
import csv
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.new_loss import predict_keypoints

class PCKEvaluator:
    def __init__(self, model, device):
        """
        Args:
            model: The FineTuner model instance (must have features_cache and forward_unfrozen_blocks).
            device: 'cuda' or 'cpu'.
        """
        self.model = model
        self.device = device
        # Use the model's configured standard size (e.g., 518 for DINOv2, 512 for DINOv3)
        self.standard_size = getattr(model, 'standard_size', 518)

    def evaluate(self, dataloader, output_path, alpha=0.1):
        """
        Runs evaluation on the dataloader and saves detailed results to CSV.
        
        Args:
            dataloader: DataLoader for the test set (batch_size=1 recommended).
            output_path: Path to save the 'test_raw_detailed.csv'.
            alpha: Threshold factor for PCK (default 0.1).
        """
        self.model.backbone.eval()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Evaluating on {len(dataloader)} pairs...")
        print(f"Standard Size for Inference: {self.standard_size}x{self.standard_size}")
        print(f"Saving per-keypoint details to: {output_path}")

        # Open CSV for writing
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header matching your requirement for detailed analysis
            writer.writerow(['pair_idx', 'src_img', 'trg_img', 'category', 'kps_idx', 
                             'pixel_error', 'bbox_size', 'threshold', 'is_correct', 'is_visible'])

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval")):
                    self._process_batch(batch, batch_idx, writer, alpha)
        
        print(f"Evaluation finished. Results saved.")
        return output_path

    def _process_batch(self, batch, batch_idx, writer, alpha):
        """
        Internal method to process a single batch using the Explicit Geometric Pipeline.
        """
        # Unpack Metadata (Handle batch_size=1 or >1)
        src_names = batch['src_name']
        trg_names = batch['trg_name']
        src_kps_raw = batch['src_kps']   # (B, N, 2)
        trg_kps_raw = batch['trg_kps']   # (B, N, 2)
        trg_bbox = batch['trg_bndbox']   # (B, 4)
        
        # Determine batch size dynamically
        B = len(src_names) if isinstance(src_names, list) else 1
        
        # Iterate through items in the batch
        for i in range(B):
            # Handle list vs string unpacking
            s_name = src_names[i] if B > 1 else src_names
            t_name = trg_names[i] if B > 1 else trg_names
            
            # Retrieve Cache
            if s_name not in self.model.features_cache or t_name not in self.model.features_cache:
                print(f"Warning: Cache missing for {s_name} or {t_name}. Skipping.")
                continue
                
            cache_src = self.model.features_cache[s_name]
            cache_trg = self.model.features_cache[t_name]
            
            src_orig_w, src_orig_h = cache_src['orig_size']
            trg_orig_w, trg_orig_h = cache_trg['orig_size']
            
            # Load Features & Forward Pass
            src_inter = cache_src['intermediate'].to(self.device)
            trg_inter = cache_trg['intermediate'].to(self.device)
            
            # Add batch dim if missing (cache stores 1, N, C usually)
            if src_inter.dim() == 2: src_inter = src_inter.unsqueeze(0)
            if trg_inter.dim() == 2: trg_inter = trg_inter.unsqueeze(0)
            
            feat1 = self.model.forward_unfrozen_blocks(src_inter)
            feat2 = self.model.forward_unfrozen_blocks(trg_inter)
            
            # Scale Source Keypoints: Original -> Standard
            # We select the specific keypoints for this image
            curr_src_kps = src_kps_raw[i] if B > 1 else src_kps_raw
            if not isinstance(curr_src_kps, torch.Tensor): 
                curr_src_kps = torch.tensor(curr_src_kps, dtype=torch.float32)
                
            src_kps_std = curr_src_kps.clone()
            src_kps_std[:, 0] *= (self.standard_size / src_orig_w)
            src_kps_std[:, 1] *= (self.standard_size / src_orig_h)
            src_kps_std = src_kps_std.to(self.device).unsqueeze(0) # (1, N, 2)
            
            # Predict (returns [-1, 1])
            pred_kps_norm = predict_keypoints(
                feat1, feat2, src_kps_std,
                src_img_size=(self.standard_size, self.standard_size),
                temperature=0.1
            )
            
            # Scale Prediction: [-1, 1] -> Target Original
            pred_kps_orig = pred_kps_norm.clone()
            pred_kps_orig[..., 0] = (pred_kps_norm[..., 0] + 1) / 2 * (trg_orig_w - 1)
            pred_kps_orig[..., 1] = (pred_kps_norm[..., 1] + 1) / 2 * (trg_orig_h - 1)
            pred_kps_orig = pred_kps_orig.squeeze(0).cpu() # (N, 2)
            
            # Compute Errors
            curr_trg_kps = trg_kps_raw[i] if B > 1 else trg_kps_raw
            if not isinstance(curr_trg_kps, torch.Tensor):
                curr_trg_kps = torch.tensor(curr_trg_kps)
                
            distances = torch.norm(pred_kps_orig - curr_trg_kps, dim=-1)
            
            # Compute Threshold (BBox)
            curr_bbox = trg_bbox[i] if B > 1 else trg_bbox
            if not isinstance(curr_bbox, torch.Tensor): curr_bbox = torch.tensor(curr_bbox)
            
            bbox_w = curr_bbox[2] - curr_bbox[0]
            bbox_h = curr_bbox[3] - curr_bbox[1]
            bbox_size = max(bbox_w, bbox_h).item()
            threshold = alpha * bbox_size
            
            # Logging to CSV
            category = batch['category'][i] if (B > 1 and 'category' in batch) else batch.get('category', 'unknown')
            if isinstance(category, list): category = category[0] # Handle weird batching cases
            
            num_kps = distances.shape[0]
            for k in range(num_kps):
                err = distances[k].item()
                is_visible = (curr_trg_kps[k, 0] > 0) and (curr_trg_kps[k, 1] > 0)
                is_correct = (err <= threshold) and is_visible
                
                if is_visible:
                    writer.writerow([
                        batch_idx, s_name, t_name, category, k, 
                        f"{err:.4f}", f"{bbox_size:.2f}", f"{threshold:.4f}", 
                        int(is_correct), int(is_visible)
                    ])

    def summarize_results(self, csv_path, alpha):
        """
        Reads the generated CSV and prints summary metrics.
        """
        import pandas as pd
        if not os.path.exists(csv_path):
            print("CSV not found.")
            return

        print("\n" + "-"*40)
        print("COMPUTING SUMMARY METRICS")
        print("-" * 40)
        
        df = pd.read_csv(csv_path)
        
        df = df[df['is_visible'] == 1]
        
        #PCK Per Keypoint (Total Correct / Total Visible)
        pck_kps = df['is_correct'].mean()
        
        # PCK Per Image (Mean of (Correct/Visible) per image)
        # Group by image pairs
        img_scores = df.groupby(['pair_idx', 'src_img'])['is_correct'].mean()
        pck_img = img_scores.mean()
        
        print(f"Total Keypoints Evaluated: {len(df)}")
        print(f"Total Images Evaluated:    {len(img_scores)}")
        print(f"\nPCK @ {alpha} (Per Keypoint):  {pck_kps:.2%}")
        print(f"PCK @ {alpha} (Per Image):     {pck_img:.2%}")
        print("-" * 40)