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

# Standard image size for DINOv3 feature extraction. 
# 528 = 16 * 33 (closest multiple of 16 to the 518 used in DINOv2)
STANDARD_SIZE = 528 

class DINOv3FeatureExtractor:
    def __init__(self, model_name='dinov3_vits16', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the DINOv3 model.
        
        Args:
            model_name (str): The DINOv3 model to load. 
                              Common options: 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.patch_size = 16
            
        print(f"Loading {model_name} on {self.device} with patch_size={self.patch_size}...")
        
        try:
            self.model = torch.hub.load('facebookresearch/dinov3', model_name).to(self.device)
        except Exception as e:
            print(f"Standard hub load failed ({e}). Trying with trust_repo=True...")
            self.model = torch.hub.load('facebookresearch/dinov3', model_name, trust_repo=True).to(self.device)
            
        self.model.eval() # Set to evaluation mode (frozen features)

    def preprocess_image(self, image_path, target_size=None):
        """Loads and preprocesses an image from path."""
        img = Image.open(image_path).convert('RGB')
        return self.preprocess_image_pil(img, target_size)

    def preprocess_image_pil(self, img, target_size=None):
        """
        Preprocesses a PIL image. Ensures dimensions are multiples of patch_size.
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # Resize logic: If target_size is provided, use it. 
        # Otherwise, ensure dimensions are divisible by patch_size.
        w, h = img.size if target_size is None else target_size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        # DINOv3 uses standard ImageNet normalization
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
        Extracts dense features from the image.
        Returns: Feature map of shape (1, C, H_patch, W_patch)
        """
        with torch.no_grad():
            features_dict = self.model.forward_features(image_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] # Shape: (B, N_patches, C)
            
            B, N, C = patch_tokens.shape
            H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
            H_patch = H_img // self.patch_size
            W_patch = W_img // self.patch_size
            
            if N != H_patch * W_patch:
                raise ValueError(f"Patch count mismatch! Expected {H_patch*W_patch}, got {N}.")
            
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)
            feature_map = torch.nn.functional.normalize(feature_map, dim=1)
            return feature_map

    def map_keypoints(self, keypoints, inverse=False):
        """Maps keypoints between image pixel coordinates and feature map coordinates."""
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, device=self.device)
            
        if inverse:
            return keypoints * self.patch_size + (self.patch_size / 2)
        else:
            return keypoints / self.patch_size


class DINOv3FineTuner:
    """
    Fine-tuning trainer for DINOv3-based semantic correspondence.
    Compatible with src/trainer.py.
    """
    def __init__(
        self,
        model_name='dinov3_vits16',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        learning_rate=1e-5,
        feature_extractor=None,
    ):
        self.standard_size = STANDARD_SIZE
        self.device = device
        self.model_name = model_name
        
        # Use provided feature extractor or create a new one
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            self.backbone = feature_extractor.model
            print(f"Using provided DINOv3FeatureExtractor")
        else:
            self.feature_extractor = DINOv3FeatureExtractor(model_name=model_name, device=device)
            self.backbone = self.feature_extractor.model
        
        self.patch_size = self.feature_extractor.patch_size
        self.embed_dim = self.feature_extractor.get_embed_dim()
        
        # Feature caching setup
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.backbone.blocks) - num_unfrozen_blocks
        self.features_cache = {}
        
        # 1. Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 2. Unfreeze the last N transformer blocks
        print(f"Total blocks: {len(self.backbone.blocks)}, Frozen: {self.num_frozen_blocks}, Unfrozen: {num_unfrozen_blocks}")
        
        blocks_to_unfreeze = list(self.backbone.blocks)[-num_unfrozen_blocks:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        # 3. Unfreeze final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            print("Unfreezing final norm layer...")
        
        # Statistics
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Optimizer
        trainable_backbone_params = [p for p in self.backbone.parameters() if p.requires_grad and p.is_leaf]
        self.optimizer = torch.optim.AdamW(
            [{'params': trainable_backbone_params, 'lr': learning_rate}],
            weight_decay=0.01
        )
        
        self.scheduler = None
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'epoch_train_losses': [], 'learning_rates': []
        }
    
    def extract_intermediate_features(self, image_tensor):
        """
        Extract output of FROZEN blocks (before unfrozen blocks).
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
        Run UNFROZEN blocks + norm with gradients.
        """
        x = intermediate_features
        
        # Run through UNFROZEN blocks
        for block in list(self.backbone.blocks)[-self.num_unfrozen_blocks:]:
            x = block(x)
        
        # Apply final norm
        x = self.backbone.norm(x)
        
        # Remove CLS token and reshape
        patch_tokens = x[:, 1:]  # (B, N, C)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return F.normalize(feature_map, dim=1)
    
    def extract_all_features(self, dataset, show_progress=True):
        """
        Pre-extract INTERMEDIATE features for all images at STANDARD_SIZE (528x528).
        """
        from tqdm import tqdm
        
        cache_dir = Path('checkpoints') / 'feature_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.model_name}_intermediate_{self.num_frozen_blocks}frozen_s{STANDARD_SIZE}.pt"
        
        if cache_file.exists():
            try:
                print(f"Loading intermediate features from {cache_file}...")
                self.features_cache = torch.load(cache_file, map_location='cpu')
                print(f"Loaded {len(self.features_cache)} cached features.")
                return self.features_cache
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}. Re-extracting...")
        
        print("Scanning images...")
        jpeg_dir = dataset.root / 'JPEGImages'
        unique_images = set()
        
        for category_dir in jpeg_dir.iterdir():
            if category_dir.is_dir():
                for img_file in category_dir.glob('*.jpg'):
                    unique_images.add(f"{category_dir.name}/{img_file.stem}")
        
        print(f"Extracting features at {STANDARD_SIZE}x{STANDARD_SIZE} for {len(unique_images)} images...")
        iterator = tqdm(unique_images) if show_progress else unique_images
        
        for img_name in iterator:
            if img_name in self.features_cache: continue
            
            img_path = dataset.root / 'JPEGImages' / f'{img_name}.jpg'
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            
            # Preprocess at STANDARD_SIZE (528)
            img_tensor = self.feature_extractor.preprocess_image_pil(img, target_size=(STANDARD_SIZE, STANDARD_SIZE))
            
            # Extract
            intermediate = self.extract_intermediate_features(img_tensor)
            
            self.features_cache[img_name] = {
                'intermediate': intermediate.cpu(),
                'orig_size': (orig_w, orig_h),
            }
        
        torch.save(self.features_cache, cache_file)
        print(f"Saved cache to {cache_file}")
        return self.features_cache
    
    def _collate_fn(self, batch):
        return batch[0] if len(batch) == 1 else batch
    
    def save_checkpoint(self, path):
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size
        }
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded: {path}")