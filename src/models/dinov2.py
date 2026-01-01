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
from .checkpoint_utils import safe_torch_load, extract_state_dict, clean_state_dict_keys, get_merged_state_dict
from src.lora_utils import apply_lora


# Standard image size for all feature extraction (divisible by patch_size=14)
# Multiple of 14 (14*37=518), used in the DINOv2 paper
STANDARD_SIZE = 518 

class DINOv2Adapter:
    def __init__(
        self,
        model_name='dinov2_vits14',
        weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        dropout=0.0,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
    ):
        """
        Initializes the DINOv2 model.
        
        Args:
            model_name (str): The DINOv2 model to load. Options: 
                              'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
            weights_path (str, optional): Path to custom .pth weights file. If None, loads pretrained from torch.hub.
            device (str): Computation device ('cuda' or 'cpu').
            num_unfrozen_blocks (int): Number of transformer blocks to unfreeze (or target with LoRA).
            dropout (float): Dropout rate for regularization (0.0 = no dropout).
            use_lora (bool): Whether to use LoRA instead of full fine-tuning.
            lora_rank (int): LoRA rank dimension.
            lora_alpha (int): LoRA scaling factor.
        """
        self.standard_size = STANDARD_SIZE
        self.device = device
        self.patch_size = 14 # DINOv2 usually uses patch size 14
        self.model_name = model_name
        
        # Setup dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout = self.dropout.to(device)
        
        # Load model architecture and weights
        if weights_path is not None:
            # Load custom weights
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            print(f"Loading {model_name} architecture (no pretrained weights)...")
            self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False).to(self.device)
            
            print(f"Loading custom weights from: {weights_path}")
            checkpoint = safe_torch_load(weights_path, map_location='cpu')
            state_dict, format_info = extract_state_dict(checkpoint)
            print(f"Loading from {format_info}")
            
            # Clean state dictionary keys
            state_dict = clean_state_dict_keys(state_dict)
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded successfully!")
        else:
            # Load pretrained model from torch.hub
            print(f"Loading {model_name} with pretrained weights from torch.hub...")
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        
        self.model.eval() # Set to evaluation mode (frozen features)
        
        # Store number of unfrozen blocks for feature caching logic
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.blocks) - num_unfrozen_blocks
        
        # 1. Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Apply LoRA or Unfreeze
        if use_lora:
            print(f"Applying LoRA (Rank={lora_rank}, Alpha={lora_alpha}) to last {num_unfrozen_blocks} blocks...")
            self.model = apply_lora(
                self.model, 
                model_type='dinov2', 
                rank=lora_rank, 
                alpha=lora_alpha, 
                num_unfrozen_blocks=num_unfrozen_blocks
            )
        else:
            # Standard Fine-tuning
            num_blocks = len(self.model.blocks)
            print(f"Total transformer blocks: {num_blocks}")
            print(f"Frozen blocks: {self.num_frozen_blocks}, Unfrozen blocks: {num_unfrozen_blocks}")
            if dropout > 0:
                print(f"Dropout rate: {dropout}")
            
            blocks_to_unfreeze = list(self.model.blocks)[-num_unfrozen_blocks:]
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
            
            # Also unfreeze the final norm layer
            if hasattr(self.model, 'norm'):
                for param in self.model.norm.parameters():
                    param.requires_grad = True
                print("Unfreezing final norm layer...")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        
        # Filter to ensure we only get leaf tensors that require grad
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.is_leaf]

    @property
    def backbone(self):
        """Helper to access the underlying backbone even if wrapped by Peft."""
        import peft
        if isinstance(self.model, peft.PeftModel):
            return self.model.base_model.model
        return self.model

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
         
        #ensure dimensions are divisible by patch_size.
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
            torch.Tensor: Features after processing through specified blocks (B, N, C)
                         where N = num_patches + 1 (includes CLS token)
            
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
            # 1. Run patch embedding
            x = self.backbone.patch_embed(image_tensor)
            
            # 2. Add CLS token
            cls_tokens = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # 3. Add position embeddings
            x = x + self.backbone.interpolate_pos_encoding(x, image_tensor.shape[2], image_tensor.shape[3])
            
            # 4. Run through blocks 0 to target_block_idx (inclusive)
            for i in range(target_block_idx + 1):
                x = self.backbone.blocks[i](x)
            
            return x
    
    def forward_from_block_features(self, block_features):
        """
        Apply norm and reshape block features for downstream tasks.
        
        This is useful when you want to use features from a specific block
        with the same post-processing as the standard pipeline.
        
        Args:
            block_features: Features from extract_features_from_block (B, N, C)
                           where N includes CLS token + patches
        
        Returns:
            torch.Tensor: Processed features (B, C, H, W) - ready for semantic correspondence
        """
        x = block_features.to(self.device)
        
        # Apply final norm
        x = self.backbone.norm(x)
        
        # Remove CLS token to get patch tokens only
        patch_tokens = x[:, 1:]  # (B, N-1, C)
        
        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Patch count {N} is not a perfect square"
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Apply dropout (only active during training)
        feature_map = self.dropout(feature_map)
        
        return feature_map
    
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
        
        # Apply dropout for regularization (only active during training)
        feature_map = self.dropout(feature_map)
        
        return feature_map
    
    def get_model_state(self):
        """Returns the dictionary containing model weights and metadata.
        
        LoRA weights are automatically merged into base weights if present,
        producing a clean state dict that loads without PEFT.
        """
        return {
            "backbone_state_dict": get_merged_state_dict(self.model),
            "embed_dim": self.backbone.embed_dim,
            "patch_size": self.patch_size,
            "model_name": self.model_name
        }
    
    def load_model_state(self, checkpoint):
        """Restores model weights from a checkpoint dictionary.
        
        Note: LoRA weights should already be merged at save time.
        """
        state_dict, format_info = extract_state_dict(checkpoint)
        print(f"Loading from {format_info}")
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Model state loaded for {self.model_name}")