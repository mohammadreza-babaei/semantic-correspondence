import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import csv 

from src.new_loss import loss as the_new_loss
from src.plotting import plot_training_history  

class Trainer():
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.device = kwargs.get('device')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_train_losses': [],
            'learning_rates': []
        }
        self.scheduler = None

    def train_step(self, batch):
        """Single training step using cached intermediate features."""
        self.model.optimizer.zero_grad()
        
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        if src_name not in self.model.features_cache or trg_name not in self.model.features_cache:
            raise RuntimeError(f"Intermediate features not cached for {src_name} or {trg_name}. "
                             "Call extract_all_features() before training.")
        
        src_intermediate = self.model.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.model.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.model.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.model.features_cache[trg_name]['orig_size']
        
        feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
        feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)
        
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        src_kps = src_kps.clone()
        src_kps[:, 0] = src_kps[:, 0] * (self.model.standard_size / src_orig_w)
        src_kps[:, 1] = src_kps[:, 1] * (self.model.standard_size / src_orig_h)
        
        trg_kps = trg_kps.clone()
        trg_kps[:, 0] = trg_kps[:, 0] * (self.model.standard_size / trg_orig_w)
        trg_kps[:, 1] = trg_kps[:, 1] * (self.model.standard_size / trg_orig_h)
        
        src_kps = src_kps.to(self.device)
        trg_kps = trg_kps.to(self.device)
        
        if src_kps.shape[-1] == 3:
            vis = (src_kps[:, 2] > 0) & (trg_kps[:, 2] > 0)
            src_kps_vis = src_kps[vis, :2]
            trg_kps_vis = trg_kps[vis, :2]
        else:
            src_kps_vis = src_kps[:, :2] if src_kps.shape[-1] > 2 else src_kps
            trg_kps_vis = trg_kps[:, :2] if trg_kps.shape[-1] > 2 else trg_kps
        
        if len(src_kps_vis) == 0:
            return 0.0
        
        src_kps_batch = src_kps_vis.unsqueeze(0)
        trg_kps_batch = trg_kps_vis.unsqueeze(0)
        
        loss = the_new_loss(
            feat1, feat2, 
            src_kps_batch, trg_kps_batch, 
            src_img_size=(self.model.standard_size, self.model.standard_size),
            trg_img_size=(self.model.standard_size, self.model.standard_size),
            temperature=10.0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), max_norm=1.0)
        self.model.optimizer.step()
        
        return loss.item()
    
    def val_step(self, batch):
        """Single validation step using cached intermediate features."""
        self.model.backbone.eval()
        
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        if src_name not in self.model.features_cache or trg_name not in self.model.features_cache:
            raise RuntimeError(f"Intermediate features not cached for {src_name} or {trg_name}. "
                             "Call extract_all_features() before validation.")
        
        src_intermediate = self.model.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.model.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.model.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.model.features_cache[trg_name]['orig_size']
        
        with torch.no_grad():
            feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
            feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)
        
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        src_kps = src_kps.clone()
        src_kps[:, 0] = src_kps[:, 0] * (self.model.standard_size / src_orig_w)
        src_kps[:, 1] = src_kps[:, 1] * (self.model.standard_size / src_orig_h)
        
        trg_kps = trg_kps.clone()
        trg_kps[:, 0] = trg_kps[:, 0] * (self.model.standard_size / trg_orig_w)
        trg_kps[:, 1] = trg_kps[:, 1] * (self.model.standard_size / trg_orig_h)
        
        src_kps = src_kps.to(self.device)
        trg_kps = trg_kps.to(self.device)
        
        if src_kps.shape[-1] == 3:
            vis = (src_kps[:, 2] > 0) & (trg_kps[:, 2] > 0)
            src_kps_vis = src_kps[vis, :2]
            trg_kps_vis = trg_kps[vis, :2]
        else:
            src_kps_vis = src_kps[:, :2] if src_kps.shape[-1] > 2 else src_kps
            trg_kps_vis = trg_kps[:, :2] if trg_kps.shape[-1] > 2 else trg_kps
        
        if len(src_kps_vis) == 0:
            return 0.0
                # Add batch dimension to keypoints: (N, 2) -> (1, N, 2)
        src_kps_batch = src_kps_vis.unsqueeze(0)
        trg_kps_batch = trg_kps_vis.unsqueeze(0)        
        with torch.no_grad():
            loss = the_new_loss(
                feat1, feat2, 
                src_kps_batch, trg_kps_batch, 
                src_img_size=(self.model.standard_size, self.model.standard_size),
                trg_img_size=(self.model.standard_size, self.model.standard_size),
                temperature=10.0
            )
        
        return loss.item()
    
    def clear_features_cache(self):
        """Clear the features cache to free memory."""
        self.model.features_cache = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, train_dataset, **kwargs):
        """
        Main training loop with CSV logging and final visualization.
        """
        val_dataset = kwargs.get('val_dataset', None)
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 1)
        log_interval = kwargs.get('log_interval', 10)
        save_path = kwargs.get('save_path')
        max_iters_per_epoch = kwargs.get('max_iters_per_epoch', None)
        
        print("\n" + "="*60)
        print("Pre-extracting features...")
        print("="*60)
        self.model.extract_all_features(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            collate_fn=self.model._collate_fn
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.model._collate_fn
            )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, T_max=epochs * len(train_loader)
        )
        
        os.makedirs(save_path, exist_ok=True)

        # CSV SETUP ------------
        csv_filename = f"training_log_e{epochs}_b{batch_size}.csv"
        csv_file_path = os.path.join(save_path, csv_filename)
                
        # Create file and write header
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
            
        print(f"Logging metrics to: {csv_file_path}")
        # CSV SETUP END ------------

        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.backbone.train()
            epoch_losses = []
            max_iters = len(train_loader) if max_iters_per_epoch is None else min(max_iters_per_epoch, len(train_loader))
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_iters:
                    break
                    
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.history['epoch_train_losses'].append(loss)
                self.history['learning_rates'].append(self.model.optimizer.param_groups[0]['lr'])
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{max_iters}] Loss: {avg_loss:.4f}")
            
            train_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            val_loss_scalar = None
            if val_loader is not None:
                self.model.backbone.eval()
                val_losses = []
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_iters:
                        break
                    val_losses.append(self.val_step(batch))
                val_loss = np.mean(val_losses)
                val_loss_scalar = val_loss 
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(f"{save_path}/best_model.pt")
            
            # write to CSV
            with open(csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                current_lr = self.model.optimizer.param_groups[0]['lr']
                val_str = f"{val_loss:.6f}" if val_loss is not None else ""
                writer.writerow([epoch + 1, f"{train_loss:.6f}", val_str, f"{current_lr:.2e}"])

            print(f"\n{'─'*40}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            print(f"{'─'*40}\n")
            
            self.model.save_checkpoint(f"{save_path}/epoch_{epoch+1}.pt")
        
        # --- FINAL PLOT ---
        print("Generating final training graph...")
        plot_training_history(self.history, save_path=f"{save_path}/final_training_curves.png")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Checkpoints and logs saved to: {save_path}")
        print(f"{'='*60}")
        
        return self.history