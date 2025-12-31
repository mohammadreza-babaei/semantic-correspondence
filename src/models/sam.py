import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry
import os
from .checkpoint_utils import safe_torch_load, extract_state_dict


# Standard image size for all feature extraction (divisible by patch_size=16)
# Reduced from 1024 for speed/memory efficiency if needed. 
# Default 1024 is native SAM resolution.
STANDARD_SIZE = 512

class SAMAdapter:
    def __init__(
        self,
        model_name='vit_b',
        weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        dropout=0.0,
        unfreeze_neck=False,
    ):
        """
        Initializes the SAM model using the official segment_anything library.
        
        Args:
            model_name (str): Model type. Options:
                              'vit_b' (Recommended for speed/memory)
                              'vit_l'
                              'vit_h'
            weights_path (str): Path to .pth weights file. Required for loading the model.
            device (str): Computation device.
            num_unfrozen_blocks (int): Number of transformer blocks to unfreeze from the end.
            dropout (float): Dropout rate for regularization (0.0 = no dropout).
            unfreeze_neck (bool): Whether to unfreeze the neck projection layers for training.
        """
        self.device = device
        self.patch_size = 16  # SAM uses a patch size of 16 for its encoder
        self.standard_size = STANDARD_SIZE
        self.model_name = model_name
        
        # Setup dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout = self.dropout.to(device)
        
        if model_name not in sam_model_registry:
            raise ValueError(f"Unknown model type: {model_name}. "
                           f"Valid options: {list(sam_model_registry.keys())}")
        
        print(f"Loading SAM {model_name} on {self.device}...")

        # --- FIX: ROBUST CHECKPOINT LOADING ---
        # If path is provided but missing, warn and switch to None (Random Init)
        if weights_path is not None and not os.path.exists(weights_path):
            print(f"Warning: SAM weights path '{weights_path}' not found. Initializing with random weights.")
            weights_path = None
        
        # weights_path can be None here -> SAM registry initializes random weights
        # We pass checkpoint=None and load manually to handle weights_only=False
        # which is required for checkpoints containing numpy scalars (like recent pytorch versions)
        self.sam_model = sam_model_registry[model_name](checkpoint=None)
        
        if weights_path is not None:
            print(f"Loading weights from {weights_path}...")
            checkpoint = safe_torch_load(weights_path, map_location=self.device)
            self.load_model_state(checkpoint)
            
        self.sam_model.to(self.device)
        self.sam_model.eval()
        # --------------------------------------
        
        # Access the image encoder (vision encoder)
        self.model = self.sam_model.image_encoder
        
        # Store number of unfrozen blocks for feature caching logic
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.blocks) - num_unfrozen_blocks
        
        # Freeze all parameters first
        for param in self.sam_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer blocks in the image encoder
        num_blocks = len(self.model.blocks)
        print(f"Total transformer blocks: {num_blocks}")
        print(f"Frozen blocks: {self.num_frozen_blocks}, Unfrozen blocks: {num_unfrozen_blocks}")
        if dropout > 0:
            print(f"Dropout rate: {dropout}")
        
        blocks_to_unfreeze = self.model.blocks[-num_unfrozen_blocks:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        # Unfreeze neck if requested
        if unfreeze_neck:
            print("Unfreezing neck layers...")
            for param in self.model.neck.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.sam_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.sam_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        self.trainable_params = [p for p in self.sam_model.parameters() if p.requires_grad and p.is_leaf]

    def preprocess_image(self, image_path, target_size=None):
        """
        Loads and preprocesses an image.
        SAM works best with 1024x1024, but can handle other sizes if divisible by 16.
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
        
    
    def extract_intermediate_features(self, image_tensor):
        """
        Extract INTERMEDIATE features - output of frozen blocks, before unfrozen blocks.
        
        Args:
            image_tensor: Preprocessed image tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Intermediate features (B, H, W, C) - input to unfrozen blocks
        """
        with torch.no_grad():
            # Run patch embedding - SAM returns (B, H, W, C)
            x = self.model.patch_embed(image_tensor)
            
            # Add positional embedding if present
            if self.model.pos_embed is not None:
                pos_embed = self.model.pos_embed
                # Check if we need to interpolate positional embeddings
                # pos_embed is (1, H_orig, W_orig, C), x is (B, H, W, C)
                if pos_embed.shape[1:3] != x.shape[1:3]:
                    # Interpolate pos_embed to match current spatial size
                    # Reshape to (1, C, H, W) for interpolation
                    pos_embed = pos_embed.permute(0, 3, 1, 2)
                    pos_embed = F.interpolate(
                        pos_embed, 
                        size=(x.shape[1], x.shape[2]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # Reshape back to (1, H, W, C)
                    pos_embed = pos_embed.permute(0, 2, 3, 1)
                x = x + pos_embed
            
            # Run through FROZEN blocks only
            for i, block in enumerate(self.model.blocks):
                if i >= self.num_frozen_blocks:
                    break
                x = block(x)
            
            return x
    
    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Run intermediate features through unfrozen blocks + neck.
        
        This method DOES track gradients for training.
        
        Args:
            intermediate_features: Output from frozen blocks (B, H, W, C)
            
        Returns:
            torch.Tensor: Feature map (B, C, H, W) - L2 normalized spatial features
        """
        # Ensure input is on the correct device
        x = intermediate_features.to(self.device)
        
        # Run through UNFROZEN blocks (input is already in B, H, W, C format)
        start_idx = self.num_frozen_blocks
        for i in range(start_idx, len(self.model.blocks)):
            x = self.model.blocks[i](x)
        
        # x is (B, H, W, C), need to reshape to (B, C, H, W) for neck
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.model.neck(x)
        
        # Apply dropout for regularization (only active during training)
        x = self.dropout(x)
        
        return x
    
    def extract_features_from_block(self, image_tensor, target_block_idx):
        """
        Extract features up to and including a specific block depth.
        
        This method is useful for analyzing which transformer blocks provide
        the best representations for semantic correspondence tasks. It extracts
        features cumulatively through blocks 0 to target_block_idx.
        
        Args:
            image_tensor: Preprocessed image tensor (B, C, H, W)
            target_block_idx: Block index to extract through (inclusive)
                             - Positive: 0-indexed (e.g., 8 = blocks 0-8)
                             - Negative: from end (e.g., -1 = all blocks, -2 = all but last)
        
        Returns:
            torch.Tensor: Features after processing through specified blocks (B, H, W, C)
            
        Example:
            # For a 12-block ViT:
            # Extract features through block 8 (processes blocks 0-8)
            features = model.extract_features_from_block(img, 8)
            
            # Extract through last block
            features = model.extract_features_from_block(img, -1)
        """
        num_blocks = len(self.model.blocks)
        
        # Handle negative indexing
        if target_block_idx < 0:
            target_block_idx = num_blocks + target_block_idx
        
        # Validate block index
        if target_block_idx < 0 or target_block_idx >= num_blocks:
            raise ValueError(
                f"Block index {target_block_idx} out of range. "
                f"Model has {num_blocks} blocks (valid range: 0-{num_blocks-1} or -{num_blocks} to -1)."
            )
        
        with torch.no_grad():
            # 1. Run patch embedding - SAM returns (B, H, W, C)
            x = self.model.patch_embed(image_tensor)
            
            # 2. Add positional embedding if present
            if self.model.pos_embed is not None:
                pos_embed = self.model.pos_embed
                # Check if we need to interpolate positional embeddings
                if pos_embed.shape[1:3] != x.shape[1:3]:
                    # Interpolate pos_embed to match current spatial size
                    pos_embed = pos_embed.permute(0, 3, 1, 2)
                    pos_embed = F.interpolate(
                        pos_embed, 
                        size=(x.shape[1], x.shape[2]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    pos_embed = pos_embed.permute(0, 2, 3, 1)
                x = x + pos_embed
            
            # 3. Run through blocks 0 to target_block_idx (inclusive)
            for i in range(target_block_idx + 1):
                x = self.model.blocks[i](x)
            
            return x

    
    def forward_from_block_features(self, block_features):
        """
        Apply neck and normalization to block features for downstream tasks.
        
        This is useful when you want to use features from a specific block
        with the same post-processing as the standard pipeline.
        
        Args:
            block_features: Features from extract_features_from_block (B, H, W, C)
        
        Returns:
            torch.Tensor: Processed features (B, C, H, W) - ready for semantic correspondence
        """
        x = block_features.to(self.device)
        
        # Reshape to (B, C, H, W) for neck
        x = x.permute(0, 3, 1, 2)
        
        # Apply neck (projection layers)
        x = self.model.neck(x)
        
        # Apply dropout (only active during training)
        x = self.dropout(x)
        
        return x
    

    def get_model_state(self):
        """Returns the dictionary containing model weights and metadata."""
        return {
            "backbone_state_dict": self.model.state_dict(),
            "sam_model_state_dict": self.sam_model.state_dict(),
            "patch_size": self.patch_size,
            "model_name": self.model_name
        }
    
    def load_model_state(self, checkpoint):
        """Restores model weights from a checkpoint dictionary.
        
        Handles two formats:
        - sam_model_state_dict: Full SAM model state (image_encoder + decoder + prompt)
        - backbone_state_dict: Just the image encoder
        """
        # Handle Trainer checkpoint format (model_state wraps everything)
        state = checkpoint
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        
        # Prefer full SAM model state dict if available
        if "sam_model_state_dict" in state:
            print(f"Loading full SAM model state...")
            self.sam_model.load_state_dict(state["sam_model_state_dict"])
        elif "backbone_state_dict" in state:
            # Fallback: load just the image encoder
            print(f"Loading backbone/encoder state only...")
            self.model.load_state_dict(state["backbone_state_dict"])
        else:
            # Plain state dict - assume it's full SAM model format
            print(f"Loading plain state dict as full SAM model...")
            self.sam_model.load_state_dict(state)
        
        print(f"Model state loaded for {self.model_name}")