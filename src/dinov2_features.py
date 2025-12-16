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

from src.loss_and_model import cal_clip_loss

class DINOv2FeatureExtractor:
    def __init__(self, model_name='dinov2_vits14', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the DINOv2 model.
        
        Args:
            model_name (str): The DINOv2 model to load. Options: 
                              'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.patch_size = 14 # DINOv2 usually uses patch size 14
        
        print(f"Loading {model_name} on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        self.model.eval() # Set to evaluation mode (frozen features)

    def preprocess_image(self, image_path, target_size=None):
        """
        Loads and preprocesses an image. Ensures dimensions are multiples of patch_size.
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize logic: If target_size is provided, use it. 
        # Otherwise, ensure dimensions are divisible by patch_size for ViT.
        w, h = img.size if target_size is None else target_size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        resize_transform = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        return resize_transform(img).unsqueeze(0).to(self.device)

    def extract_features(self, image_tensor):
        """
        Extracts dense features from the image.
        
        Returns:
            torch.Tensor: Feature map of shape (1, C, H_patch, W_patch)
                          where H_patch = H_img // 14, W_patch = W_img // 14
        """
        with torch.no_grad():
            # DINOv2 forward_features returns a dict. 
            # 'x_norm_patchtokens' contains the patch features after the last block & norm.
            features_dict = self.model.forward_features(image_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] # Shape: (B, N_patches, C)
            
            # Reshape tokens back to spatial grid
            B, N, C = patch_tokens.shape
            H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
            H_patch = H_img // self.patch_size
            W_patch = W_img // self.patch_size
            
            # Sanity check: ensure patch count matches dimensions
            assert N == H_patch * W_patch, "Patch count mismatch!"
            
            # Reshape to (B, C, H, W) for easier cosine similarity calculation later
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)
            
            # L2 Normalize features (critical for Cosine Similarity) 
            feature_map = torch.nn.functional.normalize(feature_map, dim=1)
            
            return feature_map

    def map_keypoints(self, keypoints, inverse=False):
        """
        Maps keypoints between image pixel coordinates and feature map coordinates.
        
        Args:
            keypoints (torch.Tensor or numpy.ndarray): Shape (N, 2) containing (x, y) coordinates.
            inverse (bool): If False (default), maps Image Pixels -> Feature Grid (divides by patch_size).
                            If True, maps Feature Grid -> Image Pixels (multiplies by patch_size).
        
        Returns:
            torch.Tensor: Mapped keypoints.
        """
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, device=self.device)
            
        if inverse:
            # Feature Grid -> Image Pixels
            # We map the center of the patch back to the pixel space
            return keypoints * self.patch_size + (self.patch_size / 2)
        else:
            # Image Pixels -> Feature Grid
            return keypoints / self.patch_size

    def fine_tune(self, checkpoint_path='checkpoints/dinov2_features/dinov2_vits14_features.pt'):
        """
        Loads pretrained features from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file containing pretrained features.
        
        Returns:
            dict: Dictionary containing the loaded features with image identifiers as keys.
        """
        print(f"Loading features from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            print(f"Loaded {len(checkpoint) - 1 if 'device' in checkpoint else len(checkpoint)} feature sets from checkpoint.")
            # Remove 'device' key if present since it's metadata
            if 'device' in checkpoint:
                del checkpoint['device']
            return checkpoint
        else:
            raise ValueError("Checkpoint format not recognized. Expected a dictionary.")



class DINOv2FineTuner:
    """Fine-tuning trainer for DINOv2-based semantic correspondence by unfreezing last layers."""
    
    def __init__(
        self,
        model_name='dinov2_vits14',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        learning_rate=1e-5,
    ):
        self.device = device
        self.patch_size = 14
        
        print(f"Loading {model_name} on {self.device}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        
        # Get embedding dimension from backbone
        self.embed_dim = self.backbone.embed_dim
        
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer blocks
        num_blocks = len(self.backbone.blocks)
        print(f"Total transformer blocks: {num_blocks}")
        print(f"Unfreezing last {num_unfrozen_blocks} blocks...")
        
        blocks_to_unfreeze = list(self.backbone.blocks)[-num_unfrozen_blocks:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        # Also unfreeze the final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            print("Unfreezing final norm layer...")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Learnable temperature parameters for contrastive loss
        # Create parameters directly on device to keep them as leaf tensors
        self.logit_scale = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))
        self.self_logit_scale = nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07))
        
        # Optimizer for unfrozen backbone parameters + temperature parameters
        # Filter to ensure we only get leaf tensors that require grad
        trainable_backbone_params = [p for p in self.backbone.parameters() if p.requires_grad and p.is_leaf]
        self.optimizer = torch.optim.AdamW(
            [
                {'params': trainable_backbone_params, 'lr': learning_rate},
                {'params': [self.logit_scale, self.self_logit_scale], 'lr': learning_rate * 10}
            ],
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_train_losses': [],
            'learning_rates': []
        }
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def preprocess_image(self, image):
        """Preprocess image ensuring dimensions are multiples of patch_size."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        w, h = image.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.BILINEAR)
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def extract_features(self, image_tensor, requires_grad=False):
        """Extract features from backbone."""
        if requires_grad:
            # Training mode - allow gradients through unfrozen layers
            features_dict = self.backbone.forward_features(image_tensor)
            patch_tokens = features_dict['x_norm_patchtokens']
        else:
            with torch.no_grad():
                features_dict = self.backbone.forward_features(image_tensor)
                patch_tokens = features_dict['x_norm_patchtokens']
        
        B, N, C = patch_tokens.shape
        H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
        H_patch = H_img // self.patch_size
        W_patch = W_img // self.patch_size
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)
        # L2 normalize features
        feature_map = F.normalize(feature_map, dim=1)
        return feature_map
    
    def forward(self, img1_tensor, img2_tensor):
        """Forward pass through backbone."""
        feat1 = self.extract_features(img1_tensor, requires_grad=True)
        feat2 = self.extract_features(img2_tensor, requires_grad=True)
        
        return feat1, feat2
    
    def train_step(self, batch):
        """Single training step."""
        self.backbone.train()
        
        # Extract data from batch
        src_img = batch['src_img']
        trg_img = batch['trg_img']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        # Handle PIL images vs tensors
        if isinstance(src_img, Image.Image):
            # Get original sizes
            src_orig_w, src_orig_h = src_img.size
            trg_orig_w, trg_orig_h = trg_img.size
            
            # Preprocess images
            src_tensor = self.preprocess_image(src_img)
            trg_tensor = self.preprocess_image(trg_img)
            
            # Get new sizes
            src_new_h, src_new_w = src_tensor.shape[2], src_tensor.shape[3]
            trg_new_h, trg_new_w = trg_tensor.shape[2], trg_tensor.shape[3]
            
            # Scale keypoints to match resized images
            if not isinstance(src_kps, torch.Tensor):
                src_kps = torch.tensor(src_kps, dtype=torch.float32)
                trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
            
            # Scale keypoints (x, y, visibility)
            if src_kps.shape[-1] >= 2:
                src_kps = src_kps.clone()
                src_kps[:, 0] = src_kps[:, 0] * (src_new_w / src_orig_w)
                src_kps[:, 1] = src_kps[:, 1] * (src_new_h / src_orig_h)
            
            if trg_kps.shape[-1] >= 2:
                trg_kps = trg_kps.clone()
                trg_kps[:, 0] = trg_kps[:, 0] * (trg_new_w / trg_orig_w)
                trg_kps[:, 1] = trg_kps[:, 1] * (trg_new_h / trg_orig_h)
        else:
            src_tensor = src_img.to(self.device)
            trg_tensor = trg_img.to(self.device)
            
            # Ensure keypoints are tensors
            if not isinstance(src_kps, torch.Tensor):
                src_kps = torch.tensor(src_kps, dtype=torch.float32)
                trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        src_kps = src_kps.to(self.device)
        trg_kps = trg_kps.to(self.device)
        
        # Forward pass
        feat1, feat2 = self.forward(src_tensor, trg_tensor)
        
        # Extract features at keypoint locations
        # Note: feat1 and feat2 may have different shapes if images have different sizes
        B1, C1, H1, W1 = feat1.shape
        B2, C2, H2, W2 = feat2.shape
        
        # Handle visibility if present
        if src_kps.shape[-1] == 3:
            vis = (src_kps[:, 2] > 0) & (trg_kps[:, 2] > 0)
            src_kps_vis = src_kps[vis, :2]
            trg_kps_vis = trg_kps[vis, :2]
        else:
            src_kps_vis = src_kps
            trg_kps_vis = trg_kps
        
        if len(src_kps_vis) == 0:
            return 0.0
        
        # Convert pixel coordinates to feature grid coordinates
        src_kps_grid = (src_kps_vis / self.patch_size).floor().long()
        trg_kps_grid = (trg_kps_vis / self.patch_size).floor().long()
        
        # Clamp coordinates to valid feature map range [0, dim-1] using correct dimensions
        src_kps_grid[:, 0] = src_kps_grid[:, 0].clamp(0, W1 - 1)
        src_kps_grid[:, 1] = src_kps_grid[:, 1].clamp(0, H1 - 1)
        trg_kps_grid[:, 0] = trg_kps_grid[:, 0].clamp(0, W2 - 1)
        trg_kps_grid[:, 1] = trg_kps_grid[:, 1].clamp(0, H2 - 1)
        
        # Extract features at keypoint locations
        feat1_flat = feat1[0].permute(1, 2, 0)  # (H, W, C)
        feat2_flat = feat2[0].permute(1, 2, 0)  # (H, W, C)
        
        src_kps_features = feat1_flat[src_kps_grid[:, 1], src_kps_grid[:, 0]]  # (N, C)
        trg_kps_features = feat2_flat[trg_kps_grid[:, 1], trg_kps_grid[:, 0]]  # (N, C)
        
        # Compute loss using cal_clip_loss
        loss = cal_clip_loss(
            src_kps_features,
            trg_kps_features,
            self.logit_scale.exp().clamp(max=100),
            self_logit_scale=self.self_logit_scale.exp().clamp(max=100)
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def val_step(self, batch):
        """Single validation step."""
        self.backbone.eval()
        
        src_img = batch['src_img']
        trg_img = batch['trg_img']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        if isinstance(src_img, Image.Image):
            src_tensor = self.preprocess_image(src_img)
            trg_tensor = self.preprocess_image(trg_img)
        else:
            src_tensor = src_img.to(self.device)
            trg_tensor = trg_img.to(self.device)
        
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        src_kps = src_kps.to(self.device)
        trg_kps = trg_kps.to(self.device)
        
        feat1 = self.extract_features(src_tensor, requires_grad=False)
        feat2 = self.extract_features(trg_tensor, requires_grad=False)
        
        # Extract features at keypoint locations
        # Note: feat1 and feat2 may have different shapes if images have different sizes
        B1, C1, H1, W1 = feat1.shape
        B2, C2, H2, W2 = feat2.shape
        
        # Handle visibility if present
        if src_kps.shape[-1] == 3:
            vis = (src_kps[:, 2] > 0) & (trg_kps[:, 2] > 0)
            src_kps_vis = src_kps[vis, :2]
            trg_kps_vis = trg_kps[vis, :2]
        else:
            src_kps_vis = src_kps
            trg_kps_vis = trg_kps
        
        if len(src_kps_vis) == 0:
            return 0.0
        
        # Convert pixel coordinates to feature grid coordinates
        src_kps_grid = (src_kps_vis / self.patch_size).floor().long()
        trg_kps_grid = (trg_kps_vis / self.patch_size).floor().long()
        
        # Clamp coordinates to valid feature map range [0, dim-1] using correct dimensions
        src_kps_grid[:, 0] = src_kps_grid[:, 0].clamp(0, W1 - 1)
        src_kps_grid[:, 1] = src_kps_grid[:, 1].clamp(0, H1 - 1)
        trg_kps_grid[:, 0] = trg_kps_grid[:, 0].clamp(0, W2 - 1)
        trg_kps_grid[:, 1] = trg_kps_grid[:, 1].clamp(0, H2 - 1)
        
        # Extract features at keypoint locations
        feat1_flat = feat1[0].permute(1, 2, 0)  # (H, W, C)
        feat2_flat = feat2[0].permute(1, 2, 0)  # (H, W, C)
        
        src_kps_features = feat1_flat[src_kps_grid[:, 1], src_kps_grid[:, 0]]  # (N, C)
        trg_kps_features = feat2_flat[trg_kps_grid[:, 1], trg_kps_grid[:, 0]]  # (N, C)
        
        # Compute loss using cal_clip_loss
        loss = cal_clip_loss(
            src_kps_features,
            trg_kps_features,
            self.logit_scale.exp().clamp(max=100),
            self_logit_scale=self.self_logit_scale.exp().clamp(max=100)
        )
        
        return loss.item()
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        epochs=10,
        batch_size=1,  # Usually 1 for correspondence due to varying keypoint counts
        log_interval=10,
        save_path='checkpoints/finetuned_dinov2',
        plot_every_epoch=True,
        max_iters_per_epoch=None
    ):
        """
        Main training loop with loss visualization.
        
        Args:
            train_dataset: Training dataset (SPair71kPairs or similar)
            val_dataset: Optional validation dataset
            epochs: Number of training epochs
            batch_size: Batch size (typically 1 for correspondence)
            log_interval: How often to log training progress
            save_path: Path to save checkpoints
            plot_every_epoch: Whether to update plots after each epoch
            max_iters_per_epoch: Maximum iterations per epoch (None for full epoch)
        
        Returns:
            history: Dictionary containing training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging, increase for speed
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(train_loader)
        )
        
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize plot
        if plot_every_epoch:
            plt.ion()  # Interactive mode
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.backbone.train()
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
                    self.optimizer.param_groups[0]['lr']
                )
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-log_interval:])
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Batch [{batch_idx+1}/{max_iters}] "
                          f"Loss: {avg_loss:.4f} "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate epoch average
            train_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                self.backbone.eval()
                val_losses = []
                for batch in val_loader:
                    val_losses.append(self.val_step(batch))
                val_loss = np.mean(val_losses)
                self.history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"{save_path}/best_model.pt")
            
            # Print epoch summary
            print(f"\n{'─'*40}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Best Val:   {best_val_loss:.4f}")
            print(f"{'─'*40}\n")
            
            # Update plots
            if plot_every_epoch:
                self._update_plots(fig, axes, epoch + 1)
                plt.pause(0.1)
            
            # Save checkpoint every epoch
            self.save_checkpoint(f"{save_path}/epoch_{epoch+1}.pt")
        
        # Final plot
        if plot_every_epoch:
            plt.ioff()
            self._update_plots(fig, axes, epochs)
            plt.savefig(f"{save_path}/training_curves.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        # Save final model
        self.save_checkpoint(f"{save_path}/final_model.pt")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        if self.history['val_loss']:
            print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_path}")
        print(f"{'='*60}")
        
        return self.history
    
    def _collate_fn(self, batch):
        """Custom collate function for SPair dataset."""
        # For batch_size=1, just return the single item
        if len(batch) == 1:
            return batch[0]
        # For larger batches, keep as list
        return batch
    
    def _update_plots(self, fig, axes, current_epoch):
        """Update training visualization plots."""
        axes[0].clear()
        axes[1].clear()
        
        # Plot 1: Epoch-level losses
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], 'b-o', 
                     label='Train Loss', linewidth=2, markersize=6)
        if self.history['val_loss']:
            axes[0].plot(epochs, self.history['val_loss'], 'r-s', 
                         label='Val Loss', linewidth=2, markersize=6)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Training Progress (Epoch {current_epoch})', fontsize=14)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0.5, max(current_epoch, 1) + 0.5)
        
        # Plot 2: Batch-level losses (smoothed)
        if len(self.history['epoch_train_losses']) > 0:
            batch_losses = self.history['epoch_train_losses']
            
            # Moving average smoothing
            window_size = min(50, len(batch_losses) // 10 + 1)
            if window_size > 1:
                smoothed = np.convolve(batch_losses, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
                x_smooth = range(window_size // 2, len(batch_losses) - window_size // 2)
                axes[1].plot(x_smooth, smoothed, 'g-', 
                             label=f'Smoothed (window={window_size})', 
                             linewidth=2, alpha=0.9)
            
            # Raw losses (semi-transparent)
            axes[1].plot(batch_losses, 'b-', alpha=0.3, 
                         label='Raw', linewidth=0.5)
            
            axes[1].set_xlabel('Batch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Batch-level Training Loss', fontsize=14)
            axes[1].legend(loc='upper right', fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'logit_scale': self.logit_scale,
            'self_logit_scale': self.self_logit_scale,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.logit_scale = checkpoint['logit_scale']
        self.self_logit_scale = checkpoint.get('self_logit_scale', 
                                                nn.Parameter(torch.ones([], device=self.device) * np.log(1 / 0.07)))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded: {path}")


def plot_training_history(history, save_path=None):
    """
    Plot training history after training is complete.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'epoch_train_losses'
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Epoch losses
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss per Epoch', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Batch losses with smoothing
    if history.get('epoch_train_losses'):
        batch_losses = history['epoch_train_losses']
        window = min(100, len(batch_losses) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
            axes[1].plot(range(len(smoothed)), smoothed, 'g-', linewidth=2, 
                         label=f'Smoothed (w={window})')
        axes[1].plot(batch_losses, 'b-', alpha=0.2, label='Raw')
        axes[1].set_xlabel('Batch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Batch-level Loss', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    if history.get('learning_rates'):
        axes[2].plot(history['learning_rates'], 'purple', linewidth=2)
        axes[2].set_xlabel('Batch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate Schedule', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


# --- Usage Example ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/workspaces/aml_semantic_correspondence')
    
    print("="*60)
    print("DINOv2 Fine-Tuning for Semantic Correspondence")
    print("="*60)
    
    # Check if SPair-71k dataset is available
    data_path = Path('/workspaces/aml_semantic_correspondence/data')
    spair_path = data_path / 'SPair-71k'
    
    if spair_path.exists():
        print("\n✓ SPair-71k dataset found!")
        
        # Import SPair dataset
        from src.spair_dataset import SPair71kPairs
        
        # Load datasets
        print("\nLoading datasets...")
        train_dataset = SPair71kPairs(root=str(data_path), split='train')
        val_dataset = SPair71kPairs(root=str(data_path), split='val')
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        
        # Initialize fine-tuner (unfreezing last 2 transformer blocks)
        print("\nInitializing DINOv2 Fine-Tuner...")
        finetuner = DINOv2FineTuner(
            model_name='dinov2_vits14',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_unfrozen_blocks=2,
            learning_rate=1e-5,
        )
        
        # Run training
        print("\nStarting training...")
        history = finetuner.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=5,
            batch_size=1,
            log_interval=50,
            save_path='checkpoints/finetuned_dinov2',
            plot_every_epoch=True
        )
        
        # Plot final results
        plot_training_history(
            history, 
            save_path='checkpoints/finetuned_dinov2/final_training_curves.png'
        )
        
    else:
        print("\n⚠ SPair-71k dataset not found. Running demo with synthetic data...")
        
        # Create synthetic demo dataset
        class DemoDataset(torch.utils.data.Dataset):
            """Synthetic dataset for demonstration."""
            def __init__(self, size=100, img_size=224):
                self.size = size
                self.img_size = img_size
                
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Create random images
                src_img = Image.fromarray(
                    np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
                )
                trg_img = Image.fromarray(
                    np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
                )
                
                # Create random keypoints (N keypoints with x, y, visibility)
                n_kps = np.random.randint(5, 15)
                src_kps = np.column_stack([
                    np.random.randint(0, self.img_size, n_kps),  # x
                    np.random.randint(0, self.img_size, n_kps),  # y
                    np.ones(n_kps)  # visibility
                ]).astype(np.float32)
                
                trg_kps = np.column_stack([
                    np.random.randint(0, self.img_size, n_kps),
                    np.random.randint(0, self.img_size, n_kps),
                    np.ones(n_kps)
                ]).astype(np.float32)
                
                return {
                    'src_img': src_img,
                    'trg_img': trg_img,
                    'src_kps': src_kps,
                    'trg_kps': trg_kps
                }
        
        # Create demo datasets
        train_dataset = DemoDataset(size=200)
        val_dataset = DemoDataset(size=50)
        
        print(f"  Demo train samples: {len(train_dataset)}")
        print(f"  Demo val samples: {len(val_dataset)}")
        
        # Initialize fine-tuner (unfreezing last 2 transformer blocks)
        print("\nInitializing DINOv2 Fine-Tuner...")
        finetuner = DINOv2FineTuner(
            model_name='dinov2_vits14',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_unfrozen_blocks=2,
            learning_rate=1e-5,
        )
        
        # Run training (fewer epochs for demo)
        print("\nStarting demo training...")
        history = finetuner.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=1,
            log_interval=20,
            save_path='checkpoints/demo_finetuned',
            plot_every_epoch=True
        )
        
        # Plot final results
        plot_training_history(
            history,
            save_path='checkpoints/demo_finetuned/final_training_curves.png'
        )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)