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
import wandb
from tqdm import tqdm


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
            'val_pck_global': [],  # Changed from single 'val_pck'
            'val_pck_window': [],  # Added window version
            'epoch_train_losses': [],
            'learning_rates': []
        }
        self.fixed_lr = kwargs.get('fixed_lr', False)
        self.scheduler = None
        self.features_cache = None

    def cache_intermediate_features(self, dataset):
        """Extract and save all intermediate features for the given dataset."""
        
        # Create cache directory and file path
        cache_dir = Path('checkpoints') / 'feature_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.model.model_name.replace('/', '_')}_intermediate_{self.model.num_frozen_blocks}frozen.pt"
        
        # Try to load all features from disk cache first
        if cache_file.exists():
            try:
                print(f"Loading intermediate features from {cache_file}...")
                self.features_cache = torch.load(cache_file, map_location='cpu')
                print(f"Loaded {len(self.features_cache)} cached intermediate features from disk")
                print(f"Features stored in RAM (CPU memory)")
                return self.features_cache
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}")
                print("Will re-extract features...")
                self.features_cache = {}
        
        # Collect ALL unique images from the JPEGImages directory (all splits)
        print("Scanning all images in JPEGImages directory...")
        jpeg_dir = dataset.root / 'JPEGImages'
        unique_images = set()
        
        # Walk through all category subdirectories
        for category_dir in jpeg_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for img_file in category_dir.glob('*.jpg'):
                    img_name = f"{category}/{img_file.stem}"
                    unique_images.add(img_name)
        
        print(f"Found {len(unique_images)} unique images across all splits")
        print(f"Extracting intermediate features at {STANDARD_SIZE}x{STANDARD_SIZE}...")
        
        # Extract features for each unique image
        iterator = tqdm(unique_images, desc="Extracting intermediate features") if show_progress else unique_images
        
        for img_name in iterator:
            if img_name in self.features_cache:
                continue
            
            # Get image path
            img_path = dataset.root / 'JPEGImages' / f'{img_name}.jpg'
            
            # Load image and get original size
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            
            # Preprocess at standard_size
            img_tensor = self.feature_extractor.preprocess_image_pil(img, target_size=(self.standard_size, self.standard_size))
            
            # Extract INTERMEDIATE features (output of frozen blocks)
            intermediate = self.extract_intermediate_features(img_tensor)
            
            # Store intermediate features in CPU memory (RAM) to save VRAM
            self.features_cache[img_name] = {
                'intermediate': intermediate.cpu(),  # Move to CPU: (1, H, W, C)
                'orig_size': (orig_w, orig_h),
            }

    def train_step(self, batch, accumulation_steps=1):
        """Single training step using cached intermediate features.
        
        Loads intermediate features (output of frozen blocks) and runs them through
        unfrozen blocks with gradient tracking for proper training.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
            accumulation_steps: Number of steps to accumulate gradients over
        """
        
        # Get image names and keypoints
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        # Get cached INTERMEDIATE features (required)
        if src_name not in self.features_cache or trg_name not in self.features_cache:
            raise RuntimeError(f"Intermediate features not cached for {src_name} or {trg_name}. "
                             "Call extract_all_features() before training.")
        
        # Load intermediate features from CPU cache and move to GPU for computation
        src_intermediate = self.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.features_cache[trg_name]['orig_size']
        
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
            temperature=0.1
        )
        
        # Backward pass (gradients flow through unfrozen blocks)
        # Scale loss for gradient accumulation
        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()
        
        # Gradient clipping remains here (gradients are accumulated)
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        
        return loss.item()
    
    def val_step(self, batch):
        """Single validation step using cached intermediate features.
        
        Uses the same loss function as training for consistent metrics.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
        """
        self.model.model.eval()
        
        # Get image names and keypoints
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        # Load intermediate features from CPU cache and move to GPU for computation
        src_intermediate = self.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.features_cache[trg_name]['orig_size']
        
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
        
       # Compute loss (Standard logic)
        with torch.no_grad():
            loss = the_new_loss(
                feat1, feat2, 
                src_kps_batch, trg_kps_batch, 
                src_img_size=(self.model.standard_size, self.model.standard_size),
                trg_img_size=(self.model.standard_size, self.model.standard_size),
                temperature=0.1
            )

        # -------------------------------------------------------------------------
        # COMPARE: GLOBAL vs WINDOW
        # -------------------------------------------------------------------------
        # Input: The SCALED keypoints (1, N, 2)
        src_kps_all_scaled = src_kps[:, :2].unsqueeze(0) 
        
        with torch.no_grad():
            # 1. Global Prediction (Baseline)
            pred_global_norm = predict_keypoints(
                feat1, feat2, src_kps_all_scaled,
                src_img_size=(self.model.standard_size, self.model.standard_size),
                temperature=0.1
            )

            # 2. Window Prediction (Refined)
            pred_window_norm = predict_keypoints_window(
                feat1, feat2, src_kps_all_scaled,
                src_img_size=(self.model.standard_size, self.model.standard_size),
                temperature=0.1
            )
            
        # Helper to denormalize and calculate PCK
        def get_pck(pred_norm):
            # Denormalize back to ORIGINAL image size
            pred_orig = pred_norm.clone()
            pred_orig[..., 0] = (pred_norm[..., 0] + 1) / 2 * (trg_orig_w - 1)
            pred_orig[..., 1] = (pred_norm[..., 1] + 1) / 2 * (trg_orig_h - 1)
            pred_orig = pred_orig.squeeze(0).cpu()
            
            score, _ = compute_pck_from_batch(batch, pred_orig, alpha=0.1)
            return score

        pck_global = get_pck(pred_global_norm)
        pck_window = get_pck(pred_window_norm)
        
        return loss.item(), pck_global, pck_window


    
    
    def clear_features_cache(self):
        """Clear the features cache to free memory."""
        self.features_cache = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        accumulation_steps = kwargs.get('accumulation_steps', 1)
        
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
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0, 
            collate_fn=self.model._collate_fn
        )

        self.cache_intermediate_features(train_loader)

        
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
        # 1. Calculate the number of steps per epoch
        if max_iters_per_epoch is not None:
            steps_per_epoch = min(max_iters_per_epoch, len(train_loader))
        else:
            steps_per_epoch = len(train_loader)

        total_steps = epochs * steps_per_epoch // accumulation_steps

        print(f"Scheduler configured for {total_steps} total steps (Cosine Decay).")


        self.optimizer = torch.optim.AdamW(
            self.model.trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # 2. Initialize scheduler with the correct total steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps
        )
        
        # Create output directory
        os.makedirs(save_path, exist_ok=True)

        # --- CSV SETUP - PCK ---
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        csv_path = metrics_dir / "val_metrics.csv"
        
        # Initialize CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss', 'pck_global', 'pck_window'])

        # CSV SETUP - TRAIN ------------
        csv_filename = f"training_log_e{epochs}_b{batch_size}.csv"
        csv_file_path = os.path.join(save_path, csv_filename)
                
        # Create file and write header
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
            
        print(f"Saving metrics (PCK) to {csv_path}")
        print(f"Logging metrics (TRAIN) to: {csv_file_path}")
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Device: {self.device}")
        print(f"Accumulation steps: {accumulation_steps}")
        print(f"Cached images: {len(self.features_cache)}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.model.train()
            epoch_losses = []
            
            # Determine iteration limit for this epoch
            max_iters = len(train_loader) if max_iters_per_epoch is None else min(max_iters_per_epoch, len(train_loader))
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_iters:
                    break
                
                # First batch of epoch (or after previous update), zero gradients
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
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    print(f"Epoch [{epoch+1}/{epochs}] "
                            f"Batch [{batch_idx+1}/{max_iters}] "
                            f"Loss: {avg_loss:.4f} "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate epoch average
            train_loss = np.mean(epoch_losses)
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

                val_loss = np.mean(val_losses)
                val_pck_g = np.mean(val_pcks_g)
                val_pck_w = np.mean(val_pcks_w)

                self.history['val_loss'].append(val_loss)
                self.history['val_pck_global'].append(val_pck_g)
                self.history['val_pck_window'].append(val_pck_w)

                if use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/pck_global": val_pck_g,
                        "val/pck_window": val_pck_w, # Compare these in WandB UI
                        "epoch": epoch
                    })

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_checkpoint(f"{save_path}/best_model.pt")

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
            
            self.model.save_checkpoint(f"{save_path}/epoch_{epoch+1}.pt")
        
        if use_wandb:
            wandb.finish()
        
        # --- FINAL PLOT ---
        print("Generating final training graph...")
        plot_training_history(self.history, save_path=f"{save_path}/final_training_curves.png")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Checkpoints and logs saved to: {save_path}")
        print(f"{'='*60}")
        
        return self.history