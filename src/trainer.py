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

from src.new_loss import loss as the_new_loss


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
        """Single training step using cached intermediate features.
        
        Loads intermediate features (output of frozen blocks) and runs them through
        unfrozen blocks with gradient tracking for proper training.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
        """
        self.model.optimizer.zero_grad()
        
        # Get image names and keypoints
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        # Get cached INTERMEDIATE features (required)
        if src_name not in self.model.features_cache or trg_name not in self.model.features_cache:
            raise RuntimeError(f"Intermediate features not cached for {src_name} or {trg_name}. "
                             "Call extract_all_features() before training.")
        
        # Load intermediate features from CPU cache and move to GPU for computation
        src_intermediate = self.model.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.model.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.model.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.model.features_cache[trg_name]['orig_size']
        
        # Run through UNFROZEN blocks (with gradients!)
        feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
        feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)
        
        # Convert keypoints to tensors
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        # Scale keypoints from original image size to standard_size
        src_kps = src_kps.clone()
        src_kps[:, 0] = src_kps[:, 0] * (self.model.standard_size / src_orig_w)
        src_kps[:, 1] = src_kps[:, 1] * (self.model.standard_size / src_orig_h)
        
        trg_kps = trg_kps.clone()
        trg_kps[:, 0] = trg_kps[:, 0] * (self.model.standard_size / trg_orig_w)
        trg_kps[:, 1] = trg_kps[:, 1] * (self.model.standard_size / trg_orig_h)
        
        src_kps = src_kps.to(self.device)
        trg_kps = trg_kps.to(self.device)
        
        # Handle visibility: skip invisible keypoints
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
        
        # Compute loss - all images are standard_size now
        loss = the_new_loss(
            feat1, feat2, 
            src_kps_batch, trg_kps_batch, 
            src_img_size=(self.model.standard_size, self.model.standard_size),
            trg_img_size=(self.model.standard_size, self.model.standard_size),
            temperature=10.0
        )
        
        # Backward pass (gradients flow through unfrozen blocks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), max_norm=1.0)
        self.model.optimizer.step()
        
        return loss.item()
    
    def val_step(self, batch):
        """Single validation step using cached intermediate features.
        
        Uses the same loss function as training for consistent metrics.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
        """
        self.model.backbone.eval()
        
        # Get image names and keypoints
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        # Get cached INTERMEDIATE features (required)
        if src_name not in self.model.features_cache or trg_name not in self.model.features_cache:
            raise RuntimeError(f"Intermediate features not cached for {src_name} or {trg_name}. "
                             "Call extract_all_features() before validation.")
        
        # Load intermediate features from CPU cache and move to GPU for computation
        src_intermediate = self.model.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.model.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.model.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.model.features_cache[trg_name]['orig_size']
        
        # Run through unfrozen blocks (no gradients in eval)
        with torch.no_grad():
            feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
            feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)
        
        # Convert keypoints to tensors
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        # Scale keypoints from original image size to standard_size
        src_kps = src_kps.clone()
        src_kps[:, 0] = src_kps[:, 0] * (self.model.standard_size / src_orig_w)
        src_kps[:, 1] = src_kps[:, 1] * (self.model.standard_size / src_orig_h)
        
        trg_kps = trg_kps.clone()
        trg_kps[:, 0] = trg_kps[:, 0] * (self.model.standard_size / trg_orig_w)
        trg_kps[:, 1] = trg_kps[:, 1] * (self.model.standard_size / trg_orig_h)
        
        src_kps = src_kps.to(self.device)
        trg_kps = trg_kps.to(self.device)
        
        # Handle visibility: skip invisible keypoints
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
        
        # Compute loss - same as training for consistent metrics
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
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress from scratch using aggregated history data.
        
        Creates a fresh plot each time without maintaining any plotting state.
        
        Args:
            save_path: Optional path to save the figure
        
        Returns:
            fig: The matplotlib figure object
        """
        # Create new figure from scratch
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Epoch-level losses
        if len(self.history['train_loss']) > 0:
            epochs = range(1, len(self.history['train_loss']) + 1)
            axes[0].plot(epochs, self.history['train_loss'], 'b-o', 
                        label='Train Loss', linewidth=2, markersize=6)
            if len(self.history['val_loss']) > 0:
                axes[0].plot(epochs, self.history['val_loss'], 'r-s', 
                            label='Val Loss', linewidth=2, markersize=6)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(f'Training Progress (Epoch {len(self.history["train_loss"])})', fontsize=14)
            axes[0].legend(loc='best', fontsize=10)
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Batch-level losses with smoothing
        if len(self.history['epoch_train_losses']) > 0:
            batch_losses = self.history['epoch_train_losses']
            
            # Moving average smoothing
            window_size = min(50, max(1, len(batch_losses) // 10))
            if window_size > 1:
                smoothed = np.convolve(batch_losses, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
                x_smooth = range(window_size // 2, window_size // 2 + len(smoothed))
                axes[1].plot(x_smooth, smoothed, 'g-', 
                            label=f'Smoothed (window={window_size})', 
                            linewidth=2, alpha=0.9)
            
            axes[1].set_xlabel('Batch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Batch-level Training Loss', fontsize=14)
            axes[1].legend(loc='best', fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate schedule
        if len(self.history['learning_rates']) > 0:
            axes[2].plot(self.history['learning_rates'], 'purple', linewidth=2)
            axes[2].set_xlabel('Batch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14)
            axes[2].grid(True, alpha=0.3)
            axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


    def train(self, train_dataset, **kwargs):
        """
        Main training loop with loss visualization.
        
        Features are always pre-extracted before training begins.
        
        Args:
            model: The model to train
            train_dataset: Training dataset (SPair71kPairs or similar)
            **kwargs:
                val_dataset: Optional validation dataset (default: None)
                epochs: Number of training epochs (default: 10)
                batch_size: Batch size (default: 1, typically 1 for correspondence)
                log_interval: How often to log training progress (default: 10)
                save_path: Path to save checkpoints (required)
                plot_every_epoch: Whether to update plots after each epoch (default: True)
                max_iters_per_epoch: Maximum iterations per epoch (default: None for full epoch)
        
        Returns:
            history: Dictionary containing training history
        """
        # Extract kwargs with defaults
        val_dataset = kwargs.get('val_dataset', None)
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 1)
        log_interval = kwargs.get('log_interval', 10)
        save_path = kwargs.get('save_path')
        plot_every_epoch = kwargs.get('plot_every_epoch', True)
        max_iters_per_epoch = kwargs.get('max_iters_per_epoch', None)
        
        # Always pre-extract features before training
        print("\n" + "="*60)
        print("Pre-extracting features...")
        print("="*60)
        self.model.extract_all_features(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging, increase for speed
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
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, T_max=epochs * len(train_loader)
        )
        
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Device: {self.device}")
        print(f"Cached images: {len(self.model.features_cache)}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.backbone.train()
            epoch_losses = []
            
            # Determine iteration limit for this epoch
            max_iters = len(train_loader) if max_iters_per_epoch is None else min(max_iters_per_epoch, len(train_loader))
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_iters:
                    break
                    
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.history['epoch_train_losses'].append(loss)
                self.history['learning_rates'].append(
                    self.model.optimizer.param_groups[0]['lr']
                )
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    print(f"Epoch [{epoch+1}/{epochs}] "
                            f"Batch [{batch_idx+1}/{max_iters}] "
                            f"Loss: {avg_loss:.4f} "
                            f"LR: {self.model.optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate epoch average
            train_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                self.model.backbone.eval()
                val_losses = []
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_iters:
                        break
                    val_losses.append(self.val_step(batch))
                val_loss = np.mean(val_losses)
                self.history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(f"{save_path}/best_model.pt")
            
            # Print epoch summary
            print(f"\n{'─'*40}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Best Val:   {best_val_loss:.4f}")
            print(f"{'─'*40}\n")
            
            # Plot training progress
            if plot_every_epoch:
                plt.close('all')  # Close any existing plots
                fig = self.plot_training_progress()
                plt.savefig(f"{save_path}/training_curves_epoch_{epoch+1}.png", 
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Save checkpoint every epoch
            self.model.save_checkpoint(f"{save_path}/epoch_{epoch+1}.pt")
        
        # Final plot
        plt.close('all')  # Close any existing plots
        fig = self.plot_training_progress(save_path=f"{save_path}/training_curves_final.png")
        plt.close(fig)
        
        # Save final model
        # self.model.save_checkpoint(f"{save_path}/final_model.pt")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        if self.history['val_loss']:
            print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_path}")
        print(f"{'='*60}")
        
        return self.history