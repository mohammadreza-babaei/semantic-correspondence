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

# Standard image size for all feature extraction (divisible by patch_size=14)
STANDARD_SIZE = 518 # Multiple of 14 (14*37=518), used in the DINOv2 paper

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
        return self.preprocess_image_pil(img, target_size)
    
    def preprocess_image_pil(self, img, target_size=None):
        """
        Preprocesses a PIL image. Ensures dimensions are multiples of patch_size.
        
        Args:
            img: PIL Image
            target_size: Optional tuple (width, height) for target size
            
        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W)
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
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
    
    def get_embed_dim(self):
        """Get the embedding dimension of the model."""
        return self.model.embed_dim

    def extract_features(self, image_tensor):
        """
        Extracts dense features from the image
        
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
    """Fine-tuning trainer for DINOv2-based semantic correspondence by unfreezing last layers.
    
    Architecture for gradient flow:
    - Pre-extracts INTERMEDIATE features from frozen blocks (cached to disk)
    - During training, only the unfrozen blocks + norm are computed with gradients
    - All images are resized to STANDARD_SIZE (518x518) for consistent feature maps
    
    This allows efficient training while maintaining proper gradient flow.
    """
    
    def __init__(
        self,
        model_name='dinov2_vits14',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        learning_rate=1e-5,
        feature_extractor=None,
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: DINOv2 model variant to use
            device: Device for computation
            num_unfrozen_blocks: Number of transformer blocks to unfreeze from the end
            learning_rate: Learning rate for training
            feature_extractor: Optional pre-initialized DINOv2FeatureExtractor. If None,
                               a new one will be created.
        """
        self.device = device
        self.model_name = model_name
        
        # Use provided feature extractor or create a new one
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            self.backbone = feature_extractor.model
            print(f"Using provided DINOv2FeatureExtractor")
        else:
            self.feature_extractor = DINOv2FeatureExtractor(model_name=model_name, device=device)
            self.backbone = self.feature_extractor.model
        
        self.patch_size = self.feature_extractor.patch_size
        self.embed_dim = self.feature_extractor.get_embed_dim()
        
        # Store number of unfrozen blocks for feature caching logic
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.backbone.blocks) - num_unfrozen_blocks
        
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer blocks
        num_blocks = len(self.backbone.blocks)
        print(f"Total transformer blocks: {num_blocks}")
        print(f"Frozen blocks: {self.num_frozen_blocks}, Unfrozen blocks: {num_unfrozen_blocks}")
        
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
        
        # Pre-extracted INTERMEDIATE features cache (output of frozen blocks)
        # These are the inputs to the unfrozen blocks
        self.features_cache = {}
    
    def extract_intermediate_features(self, image_tensor):
        """
        Extract INTERMEDIATE features - output of frozen blocks, before unfrozen blocks.
        
        Args:
            image_tensor: Preprocessed image tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Intermediate token features (B, N, C) - input to unfrozen blocks
        """
        with torch.no_grad():
            # Run patch embedding
            x = self.backbone.patch_embed(image_tensor)
            
            # Add CLS token
            cls_tokens = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Add position embeddings
            x = x + self.backbone.interpolate_pos_encoding(x, image_tensor.shape[2], image_tensor.shape[3])
            
            # Run through FROZEN blocks only
            for i, block in enumerate(self.backbone.blocks):
                if i >= self.num_frozen_blocks:
                    break
                x = block(x)
            
            return x
    
    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Run intermediate features through unfrozen blocks + norm.
        
        This method DOES track gradients for training.
        
        Args:
            intermediate_features: Output from frozen blocks (B, N, C)
            
        Returns:
            torch.Tensor: Feature map (B, C, H, W) - L2 normalized spatial features
        """
        x = intermediate_features
        
        # Run through UNFROZEN blocks
        for block in list(self.backbone.blocks)[-self.num_unfrozen_blocks:]:
            x = block(x)
        
        # Apply final norm
        x = self.backbone.norm(x)
        
        # Remove CLS token to get patch tokens only
        patch_tokens = x[:, 1:]  # (B, N, C)
        
        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # For 518x518 input with patch_size=14: 60x60
        assert H * W == N, f"Patch count {N} is not a perfect square"
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # L2 normalize features
        feature_map = F.normalize(feature_map, dim=1)
        
        return feature_map
    
    def extract_all_features(self, dataset, show_progress=True):
        """
        Pre-extract INTERMEDIATE features for all images at STANDARD_SIZE (518x518).
        
        Intermediate features are the output of frozen blocks (before unfrozen blocks).
        During training, only unfrozen blocks are computed with gradients.
        
        Args:
            dataset: SPair71kPairs dataset (any split, will extract from all JPEGImages)
            show_progress: Whether to show progress bar
            
        Returns:
            dict: Dictionary mapping image names to intermediate feature info
        """
        from tqdm import tqdm
        
        # Create cache directory and file path
        # Include num_frozen_blocks in filename since intermediate features depend on it
        cache_dir = Path('checkpoints') / 'feature_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.model_name}_intermediate_{self.num_frozen_blocks}frozen.pt"
        
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
            
            # Preprocess at STANDARD_SIZE (518x518)
            img_tensor = self.feature_extractor.preprocess_image_pil(img, target_size=(STANDARD_SIZE, STANDARD_SIZE))
            
            # Extract INTERMEDIATE features (output of frozen blocks)
            intermediate = self.extract_intermediate_features(img_tensor)
            
            # Store intermediate features in CPU memory (RAM) to save VRAM
            self.features_cache[img_name] = {
                'intermediate': intermediate.cpu(),  # Move to CPU: (1, N+1, C) including CLS token
                'orig_size': (orig_w, orig_h),
            }
        
        # Save all features to disk in a single file
        try:
            print(f"Saving {len(self.features_cache)} intermediate features to {cache_file}...")
            torch.save(self.features_cache, cache_file)
            print("Features saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save cache file: {e}")
        
        print(f"Cached intermediate features for {len(self.features_cache)} images")
        return self.features_cache
    
    def clear_features_cache(self):
        """Clear the features cache to free memory."""
        self.features_cache = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    def train_step(self, batch):
        """Single training step using cached intermediate features.
        
        Loads intermediate features (output of frozen blocks) and runs them through
        unfrozen blocks with gradient tracking for proper training.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
        """
        self.backbone.train()
        self.optimizer.zero_grad()
        
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
        feat1 = self.forward_unfrozen_blocks(src_intermediate)
        feat2 = self.forward_unfrozen_blocks(trg_intermediate)
        
        # Convert keypoints to tensors
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        # Scale keypoints from original image size to STANDARD_SIZE
        src_kps = src_kps.clone()
        src_kps[:, 0] = src_kps[:, 0] * (STANDARD_SIZE / src_orig_w)
        src_kps[:, 1] = src_kps[:, 1] * (STANDARD_SIZE / src_orig_h)
        
        trg_kps = trg_kps.clone()
        trg_kps[:, 0] = trg_kps[:, 0] * (STANDARD_SIZE / trg_orig_w)
        trg_kps[:, 1] = trg_kps[:, 1] * (STANDARD_SIZE / trg_orig_h)
        
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
        
        # Compute loss - all images are STANDARD_SIZE now
        loss = the_new_loss(
            feat1, feat2, 
            src_kps_batch, trg_kps_batch, 
            src_img_size=(STANDARD_SIZE, STANDARD_SIZE),
            trg_img_size=(STANDARD_SIZE, STANDARD_SIZE),
            temperature=10.0
        )
        
        # Backward pass (gradients flow through unfrozen blocks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def val_step(self, batch):
        """Single validation step using cached intermediate features.
        
        Uses the same loss function as training for consistent metrics.
        
        Args:
            batch: Dictionary containing 'src_name', 'trg_name', 'src_kps', 'trg_kps'
        """
        self.backbone.eval()
        
        # Get image names and keypoints
        src_name = batch['src_name']
        trg_name = batch['trg_name']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        
        # Get cached INTERMEDIATE features (required)
        if src_name not in self.features_cache or trg_name not in self.features_cache:
            raise RuntimeError(f"Intermediate features not cached for {src_name} or {trg_name}. "
                             "Call extract_all_features() before validation.")
        
        # Load intermediate features from CPU cache and move to GPU for computation
        src_intermediate = self.features_cache[src_name]['intermediate'].to(self.device)
        trg_intermediate = self.features_cache[trg_name]['intermediate'].to(self.device)
        src_orig_w, src_orig_h = self.features_cache[src_name]['orig_size']
        trg_orig_w, trg_orig_h = self.features_cache[trg_name]['orig_size']
        
        # Run through unfrozen blocks (no gradients in eval)
        with torch.no_grad():
            feat1 = self.forward_unfrozen_blocks(src_intermediate)
            feat2 = self.forward_unfrozen_blocks(trg_intermediate)
        
        # Convert keypoints to tensors
        if not isinstance(src_kps, torch.Tensor):
            src_kps = torch.tensor(src_kps, dtype=torch.float32)
            trg_kps = torch.tensor(trg_kps, dtype=torch.float32)
        
        # Scale keypoints from original image size to STANDARD_SIZE
        src_kps = src_kps.clone()
        src_kps[:, 0] = src_kps[:, 0] * (STANDARD_SIZE / src_orig_w)
        src_kps[:, 1] = src_kps[:, 1] * (STANDARD_SIZE / src_orig_h)
        
        trg_kps = trg_kps.clone()
        trg_kps[:, 0] = trg_kps[:, 0] * (STANDARD_SIZE / trg_orig_w)
        trg_kps[:, 1] = trg_kps[:, 1] * (STANDARD_SIZE / trg_orig_h)
        
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
                src_img_size=(STANDARD_SIZE, STANDARD_SIZE),
                trg_img_size=(STANDARD_SIZE, STANDARD_SIZE),
                temperature=10.0
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
        
        Features are always pre-extracted before training begins.
        
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
        # Always pre-extract features before training
        print("\n" + "="*60)
        print("Pre-extracting features...")
        print("="*60)
        self.extract_all_features(train_dataset)
        
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
        print(f"Cached images: {len(self.features_cache)}")
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
                x_smooth = range(window_size // 2, window_size // 2 + len(smoothed))
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