import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
import csv
import wandb
import random
from tqdm import tqdm
from src.evaluator import PCKEvaluator
from src.cached_features_dataset import CachedFeaturesDataset
import gc


from src.new_loss import loss as the_new_loss, predict_keypoints, predict_keypoints_window
from src.pck import compute_pck_from_batch

from src.plotting import plot_training_history  


class Trainer():
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.device = kwargs.get('device')
        self.learning_rate = kwargs.get('learning_rate')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pck_global': [],  # Global softmax  
            'val_pck_window': [],  # inference technique softmax windowed version
            'epoch_train_losses': [],
            'learning_rates': []
        }
        self.fixed_lr = kwargs.get('fixed_lr', False)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.feature_reg = kwargs.get('feature_reg', 0.0)
        self.temperature = kwargs.get('temperature', 0.02)
        self.scheduler = None
        self.cached_dataset = None  # Set in train()

        # Initialize Evaluator for validation consistency
        self.evaluator = PCKEvaluator(self, self.device, temperature=self.temperature)
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def save_checkpoint(self, path, epoch, best_val_loss):
        """Save full training checkpoint for resumption."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.get_model_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_loss': best_val_loss,
            'learning_rate': self.learning_rate,
            'fixed_lr': self.fixed_lr
        }
        torch.save(checkpoint, path)
        print(f"Full checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load full training checkpoint and restore state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore model
        self.model.load_model_state(checkpoint['model_state'])
        
        # Restore history and metrics
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_epoch = checkpoint.get('epoch', 0)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.fixed_lr = checkpoint.get('fixed_lr', self.fixed_lr)
        
        # We don't restore optimizer/scheduler here because they need the model parameters
        # which were just updated, and they are typically initialized in train().
        # We store their states to be applied after initialization.
        self._deferred_optimizer_state = checkpoint.get('optimizer_state_dict')
        self._deferred_scheduler_state = checkpoint.get('scheduler_state_dict')
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch + 1}")
        return self.start_epoch

    def cache_intermediate_features(self, dataset, num_augmentations=0):
        """Pre-extract and cache features for a dataset using CachedFeaturesDataset.
        
        This handles the disk caching of intermediate features (outputs of frozen blocks)
        to avoid redundant extraction during training/evaluation.
        """
        self.cached_dataset = CachedFeaturesDataset(
            base_dataset=dataset,
            model=self.model,
            num_augmentations=num_augmentations
        )
        self.cached_dataset.cache_features()


    def _prepare_feature_pair(self, src_name, trg_name, aug_idx_src=None, aug_idx_trg=None, requires_grad=True):
        """Load cached features and run through unfrozen blocks.
        
        Args:
            src_name: Source image name
            trg_name: Target image name  
            aug_idx_src: Optional augmentation index for source
            aug_idx_trg: Optional augmentation index for target
            requires_grad: If True, enable gradients for training
            
        Returns:
            tuple: (feat1, feat2, src_orig_size, trg_orig_size)
        """
        src_data = self.cached_dataset.load_features(src_name, aug_idx=aug_idx_src)
        trg_data = self.cached_dataset.load_features(trg_name, aug_idx=aug_idx_trg)
        
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
        
        if requires_grad:
            feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
            feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)
        else:
            with torch.no_grad():
                feat1 = self.model.forward_unfrozen_blocks(src_intermediate)
                feat2 = self.model.forward_unfrozen_blocks(trg_intermediate)
        
        return feat1, feat2, src_data['orig_size'], trg_data['orig_size']

    def _prepare_keypoints(self, src_kps, trg_kps, src_orig_size, trg_orig_size, filter_visibility=True):
        """Convert, scale to standard_size, and optionally filter by visibility.
        
        Args:
            src_kps, trg_kps: Raw keypoints (N, 2) or (N, 3) with visibility
            src_orig_size, trg_orig_size: (W, H) tuples
            filter_visibility: If True, remove invisible keypoints
            
        Returns:
            tuple: (src_kps_scaled, trg_kps_scaled) in standard_size coordinates
                   Both are (M, 2) tensors where M <= N if filtered
        """
        # src_kps and trg_kps are expected to be torch.Tensor with shape (N, 2) or (N, 3)
        
        src_orig_w, src_orig_h = src_orig_size
        trg_orig_w, trg_orig_h = trg_orig_size
        
        # Scale to standard_size
        src_kps_std = src_kps.clone()
        src_kps_std[:, 0] *= (self.model.standard_size / src_orig_w)
        src_kps_std[:, 1] *= (self.model.standard_size / src_orig_h)
        
        trg_kps_std = trg_kps.clone()
        trg_kps_std[:, 0] *= (self.model.standard_size / trg_orig_w)
        trg_kps_std[:, 1] *= (self.model.standard_size / trg_orig_h)
        
        # Filter visibility
        if filter_visibility and src_kps_std.shape[-1] == 3:
            vis = (src_kps_std[:, 2] > 0) & (trg_kps_std[:, 2] > 0)
            src_kps_std = src_kps_std[vis, :2]
            trg_kps_std = trg_kps_std[vis, :2]
        else:
            src_kps_std = src_kps_std[:, :2]
            trg_kps_std = trg_kps_std[:, :2]
        
        return src_kps_std.to(self.device), trg_kps_std.to(self.device)

    def train_step(self, batch, accumulation_steps=1):
        """Single training step using cached intermediate features.
        
        Loads intermediate features (output of frozen blocks) and runs them through
        unfrozen blocks with gradient tracking for proper training.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
            accumulation_steps: Number of steps to accumulate gradients over
        """
        
        # Get image names
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        
        # Randomly decide whether to use augmented features (50% chance each)
        src_aug_idx = None
        trg_aug_idx = None
        if hasattr(self, 'cached_dataset') and self.cached_dataset and self.cached_dataset.num_augmentations > 0:
            if random.random() < 0.5:
                src_aug_idx = random.randint(0, self.cached_dataset.num_augmentations - 1)
            if random.random() < 0.5:
                trg_aug_idx = random.randint(0, self.cached_dataset.num_augmentations - 1)
        
        # Load features and run forward pass (with gradients)
        feat1, feat2, src_orig_size, trg_orig_size = self._prepare_feature_pair(
            src_name, trg_name, src_aug_idx, trg_aug_idx, requires_grad=True
        )
        
        # Prepare keypoints (scale and filter visibility)
        src_kps_vis, trg_kps_vis = self._prepare_keypoints(
            batch['src_kps'], batch['trg_kps'], src_orig_size, trg_orig_size
        )
        
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
            temperature=self.temperature
        )
        
        # Add L2 feature regularization if enabled
        if self.feature_reg > 0:
            feat_reg_loss = self.feature_reg * (feat1.pow(2).mean() + feat2.pow(2).mean())
            loss = loss + feat_reg_loss
        
        # Backward pass (gradients flow through unfrozen blocks)
        # Scale loss for gradient accumulation
        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()
        
        # Gradient clipping remains here (gradients are accumulated)
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        
        return loss.item()
    

    def val_step(self, batch):
        self.model.model.eval()
        
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        
        # Load features (no gradients needed)
        feat1, feat2, src_orig_size, trg_orig_size = self._prepare_feature_pair(
            src_name, trg_name, requires_grad=False
        )
        
        # Prepare keypoints (scale and filter visibility)
        src_kps_vis, trg_kps_vis = self._prepare_keypoints(
            batch['src_kps'], batch['trg_kps'], src_orig_size, trg_orig_size
        )
        
        # 1. Compute Loss
        with torch.no_grad():
            if len(src_kps_vis) > 0:
                loss = the_new_loss(
                    feat1, feat2, 
                    src_kps_vis.unsqueeze(0), trg_kps_vis.unsqueeze(0), 
                    src_img_size=(self.model.standard_size, self.model.standard_size),
                    trg_img_size=(self.model.standard_size, self.model.standard_size),
                    temperature=self.temperature
                )
                loss_val = loss.item()
            else:
                loss_val = 0.0

        # 2. Compute PCK using Evaluator
        # Need unfiltered keypoints for PCK (evaluator handles visibility via batch)
        src_kps_all, _ = self._prepare_keypoints(
            batch['src_kps'], batch['trg_kps'], src_orig_size, trg_orig_size, 
            filter_visibility=False
        )
        trg_sizes = [trg_orig_size]
        
        pck_global, pck_window = self.evaluator.compute_metrics_with_sizes(
            feat1, feat2, src_kps_all, batch, trg_sizes, alpha=0.1
        )
        
        return loss_val, pck_global, pck_window


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
                max_iters_per_epoch: Maximum iterations per epoch (default: None for full epoch)
                accumulation_steps: Number of steps to accumulate gradients (default: 1)
                shuffle: Whether to shuffle the training dataset (default: True)
        
        Returns:
            history: Dictionary containing training history
        """
        # Extract kwargs with defaults
        val_dataset = kwargs.get('val_dataset', None)
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 1)
        log_interval = kwargs.get('log_interval', 10)
        save_path = kwargs.get('save_path')
        max_iters_per_epoch = kwargs.get('max_iters_per_epoch', None)
        accumulation_steps = kwargs.get('accumulation_steps', 1)
        shuffle = kwargs.get('shuffle', True)
        num_augmentations = kwargs.get('num_augmentations', 3)
        plot_every_epoch = kwargs.get('plot_every_epoch', True)
        save_checkpoints = kwargs.get('save_checkpoints', True)
        
        # WandB Setup
        use_wandb = kwargs.get('use_wandb', False)
        wandb_project = kwargs.get('wandb_project', 'semantic_correspondence')
        wandb_run_name = kwargs.get('wandb_run_name', None)

        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model_name": self.model.model_name,
                    "num_unfrozen_blocks": self.model.num_unfrozen_blocks,
                    "learning_rate": self.learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "device": str(self.device)
                }
            )
        
        # Always pre-extract features before training
        print("\n" + "="*60)
        print("Pre-extracting features...")
        print("="*60)
        
        self.cache_intermediate_features(train_dataset, num_augmentations=num_augmentations)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0, 
            collate_fn=self._collate_fn
        )

        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self._collate_fn
            )
        
        # Setup scheduler
        # Calculate the number of steps per epoch
        if max_iters_per_epoch is not None:
            steps_per_epoch = min(max_iters_per_epoch, len(train_loader))
        else:
            steps_per_epoch = len(train_loader)

        total_steps = epochs * steps_per_epoch // accumulation_steps

        print(f"Scheduler configured for {total_steps} total steps (Cosine Decay).")


        self.optimizer = torch.optim.AdamW(
            self.model.trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Initialize scheduler with the correct total steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps
        )
        
        # Restore optimizer and scheduler states if resuming
        if hasattr(self, '_deferred_optimizer_state') and self._deferred_optimizer_state:
            self.optimizer.load_state_dict(self._deferred_optimizer_state)
            print("Optimizer state restored.")
            del self._deferred_optimizer_state
            
        if hasattr(self, '_deferred_scheduler_state') and self._deferred_scheduler_state and self.scheduler:
            self.scheduler.load_state_dict(self._deferred_scheduler_state)
            print("Scheduler state restored.")
            del self._deferred_scheduler_state

        # Create output directory
        os.makedirs(save_path, exist_ok=True)

        # CSV SETUP for PCK 
        model_name_clean = self.model.model_name.replace('/', '_')
        csv_filename = f"val_metrics_{model_name_clean}_e{epochs}_b{batch_size}.csv"
        csv_path = Path(save_path) / csv_filename
        
        # Initialize or append CSV
        file_mode = 'a' if self.start_epoch > 0 and csv_path.exists() else 'w'
        with open(csv_path, file_mode, newline='') as f:
            writer = csv.writer(f)
            if file_mode == 'w':
                writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss', 'pck_global', 'pck_window'])

        # CSV SETUP for TRAIN 
        csv_filename = f"training_log_e{epochs}_b{batch_size}.csv"
        csv_file_path = os.path.join(save_path, csv_filename)
        
        # Initialize or append CSV
        file_mode = 'a' if self.start_epoch > 0 and os.path.exists(csv_file_path) else 'w'
        with open(csv_file_path, file_mode, newline='') as f:
            writer = csv.writer(f)
            if file_mode == 'w':
                writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
            
        print(f"Saving metrics (PCK) to {csv_path}")
        print(f"Logging metrics (TRAIN) to: {csv_file_path}")
        
        print(f"\n{'='*60}")
        print(f"Starting Training{' (Resumed)' if self.start_epoch > 0 else ''}")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Current Epoch: {self.start_epoch + 1}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Device: {self.device}")
        print(f"Accumulation steps: {accumulation_steps}")
        cached_count = len(list(self.cached_dataset.cache_dir.glob('*.pt'))) if self.cached_dataset else 0
        print(f"Cached images: {cached_count}")
        print(f"{'='*60}\n")
        
        best_val_loss = self.best_val_loss
        best_val_pck = getattr(self, 'best_val_pck', 0.0)
        
        for epoch in range(self.start_epoch, epochs):
            # Training phase
            self.model.model.train()
            epoch_losses = []
            
            # Determine iteration limit for this epoch
            max_iters = len(train_loader) if max_iters_per_epoch is None else min(max_iters_per_epoch, len(train_loader))
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_iters:
                    break
                
                # First batch of epoch, zero gradients
                if (batch_idx % accumulation_steps == 0):
                    self.optimizer.zero_grad()

                loss = self.train_step(batch, accumulation_steps=accumulation_steps)
                epoch_losses.append(loss)
                self.history['epoch_train_losses'].append(loss)
                self.history['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )

                # Step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler is not None and not self.fixed_lr:
                        self.scheduler.step()

                if use_wandb:
                    wandb.log({
                        "train/batch_loss": loss,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch
                    })
                
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = torch.tensor(epoch_losses[-log_interval:], dtype=torch.float32).mean().item()
                    print(f"Epoch [{epoch+1}/{epochs}] "
                            f"Batch [{batch_idx+1}/{max_iters}] "
                            f"Loss: {avg_loss:.4f} "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate epoch average
            train_loss = torch.tensor(epoch_losses, dtype=torch.float32).mean().item()
            self.history['train_loss'].append(train_loss)
            
            if use_wandb:
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "epoch": epoch
                })
            
            # Validation phase
            val_loss = None
            val_pck_g = None
            val_pck_w = None
            
            if val_loader is not None:
                self.model.model.eval()
                val_losses = []
                val_pcks_g = [] # Global
                val_pcks_w = [] # Window

                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_iters:
                        break

                    # Unpack 3 values now
                    v_loss, v_pck_g, v_pck_w = self.val_step(batch)

                    val_losses.append(v_loss)
                    val_pcks_g.append(v_pck_g)
                    val_pcks_w.append(v_pck_w)

                val_loss = torch.tensor(val_losses, dtype=torch.float32).mean().item() if val_losses else 0.0
                val_pck_g = torch.tensor(val_pcks_g, dtype=torch.float32).mean().item() if val_pcks_g else 0.0
                val_pck_w = torch.tensor(val_pcks_w, dtype=torch.float32).mean().item() if val_pcks_w else 0.0

                self.history['val_loss'].append(val_loss)
                self.history['val_pck_global'].append(val_pck_g)
                self.history['val_pck_window'].append(val_pck_w)

                if use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/pck_global": val_pck_g,
                        "val/pck_window": val_pck_w, 
                        "epoch": epoch
                    })

                # Save best model by validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_checkpoints:
                        self.save_checkpoint(f"{save_path}/best_model_loss.pt", epoch, best_val_loss)
                    print(f"New best val loss: {val_loss:.4f}")
                
                # Save best model by PCK (use window PCK as primary metric)
                if val_pck_w > best_val_pck:
                    best_val_pck = val_pck_w
                    self.best_val_pck = best_val_pck
                    if save_checkpoints:
                        self.save_checkpoint(f"{save_path}/best_model_pck.pt", epoch, best_val_loss)
                    print(f"New best PCK (window): {val_pck_w:.4f}")

                if plot_every_epoch:
                    plot_training_history(self.history, save_path=f"{save_path}/training_curves_{epoch+1}.png")

                # Write to CSV - PCK
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    counter_batch = 1
                    for loss, pg, pw in zip(val_losses, val_pcks_g, val_pcks_w):
                        writer.writerow([epoch + 1, counter_batch, train_loss, loss, pg, pw])
                        counter_batch += 1
            
            # write to CSV - TRAIN
            with open(csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                current_lr = self.optimizer.param_groups[0]['lr']
                val_str = f"{val_loss:.6f}" if val_loss is not None else ""
                writer.writerow([epoch + 1, f"{train_loss:.6f}", val_str, f"{current_lr:.2e}"])

            print(f"\n{'─'*40}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  PCK Global: {val_pck_g:.4f}")
                print(f"  PCK Window: {val_pck_w:.4f}")
            print(f"{'─'*40}\n")
            
            if save_checkpoints:
                self.save_checkpoint(f"{save_path}/epoch_{epoch+1}.pt", epoch + 1, best_val_loss)
                
                # Cleanup: keep only last 3 epoch checkpoints
                epoch_files = sorted(Path(save_path).glob('epoch_*.pt'), 
                                     key=lambda x: int(x.stem.split('_')[1]))
                for old_ckpt in epoch_files[:-3]:
                    old_ckpt.unlink()
                    print(f"Removed old checkpoint: {old_ckpt.name}")
        
        if use_wandb:
            wandb.finish()
        
        # FINAL PLOT
        print("Generating final training graph...")
        plot_training_history(self.history, save_path=f"{save_path}/final_training_curves.png")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Checkpoints and logs saved to: {save_path}")
        print(f"{'='*60}")
        
        return self.history

    def _collate_fn(self, batch):
        return batch[0] if len(batch) == 1 else batch