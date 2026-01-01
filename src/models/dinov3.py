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
from .checkpoint_utils import safe_torch_load, extract_state_dict, clean_state_dict_keys
from src.lora_utils import apply_lora

# Standard image size for DINOv3 feature extraction. 
# 512 = 16 * 32 (closest multiple of 16 to the 518 used in DINOv2)
STANDARD_SIZE = 512 
script_dir = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(script_dir, "model.safetensors")
MODEL_NAME = "dinov3_vits16"                   
REPO_SOURCE = "facebookresearch/dinov3" 

class DINOv3Adapter:
    def __init__(
        self,
        model_name='dinov3_vits16',
        weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        dropout=0.0,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
    ):
        """
        Initializes the DINOv3 model.
        
        Args:
            model_name (str): The DINOv3 model to load (e.g., 'dinov3_vits16').
            weights_path (str, optional): Path to local .pth file. If None, downloads from Hub.
            device (str): Computation device.
            num_unfrozen_blocks (int): Number of transformer blocks to unfreeze (or target with LoRA).
            dropout (float): Dropout rate for regularization (0.0 = no dropout).
            use_lora (bool): Whether to use LoRA instead of full fine-tuning.
            lora_rank (int): LoRA rank dimension.
            lora_alpha (int): LoRA scaling factor.
        """
        self.patch_size = 16
        self.standard_size = STANDARD_SIZE
        self.device = device
        self.model_name = model_name
        
        # Setup dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout = self.dropout.to(device)

        print(f"Loading {model_name} architecture...")
        self.model = torch.hub.load(REPO_SOURCE, model_name, pretrained=False).to(self.device)

        # Load weights if path provided
        if weights_path is not None:
            if os.path.exists(weights_path):
                print(f"Loading weights from: {weights_path}")
                checkpoint = safe_torch_load(weights_path, map_location='cpu')
                state_dict, format_info = extract_state_dict(checkpoint)
                print(f"Loading from {format_info}")
                
                # Clean keys and load
                state_dict = clean_state_dict_keys(state_dict)
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f"Weights loaded successfully!")
            else:
                print(f"Warning: Weights path '{weights_path}' not found. Initializing with random weights.")
        else:
            print("No weights_path provided. Initializing with random/base weights.")
        # ------------------------------------------------
        
        # Feature caching setup
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.blocks) - num_unfrozen_blocks
        
        # 1. Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Apply LoRA or Unfreeze
        if use_lora:
            print(f"Applying LoRA (Rank={lora_rank}, Alpha={lora_alpha}) to last {num_unfrozen_blocks} blocks...")
            # apply_lora wraps the model and handles target modules
            self.model = apply_lora(
                self.model, 
                model_type='dinov3', 
                rank=lora_rank, 
                alpha=lora_alpha, 
                num_unfrozen_blocks=num_unfrozen_blocks
            )
        else:
            # Standard Fine-tuning: Unfreeze the last N transformer blocks
            print(f"Total blocks: {len(self.model.blocks)}, Frozen: {self.num_frozen_blocks}, Unfrozen: {num_unfrozen_blocks}")
            if dropout > 0:
                print(f"Dropout rate: {dropout}")
            
            blocks_to_unfreeze = list(self.model.blocks)[-num_unfrozen_blocks:]
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
            
            # 3. Unfreeze final norm layer (if not using LoRA)
            if hasattr(self.model, 'norm'):
                for param in self.model.norm.parameters():
                    param.requires_grad = True
                print("Unfreezing final norm layer...")
        
        # Statistics
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.is_leaf]
        
        print(f"Initializing {model_name} on {self.device}...")

        self.model.eval() # Set to evaluation mode

    @property
    def backbone(self):
        """Helper to access the underlying backbone even if wrapped by Peft."""
        import peft
        if isinstance(self.model, peft.PeftModel):
            return self.model.base_model.model
        return self.model

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
    
    def extract_intermediate_features(self, image_tensor):
        """
        Extract output of FROZEN blocks (before unfrozen blocks).
        
        DINOv3 uses RoPE (Rotary Position Embeddings) which must be passed to each block.
        """
        with torch.no_grad():
            # Step 1: Use DINOv3's prepare_tokens_with_masks for proper tokenization
            # This handles patch embedding, CLS token, and storage tokens
            x, seq_shape = self.backbone.prepare_tokens_with_masks(image_tensor)
            
            # Step 2: Generate RoPE embeddings for this spatial shape
            rope = self.backbone.rope_embed(H=seq_shape[0], W=seq_shape[1])
            
            # Step 3: Run through FROZEN blocks only, passing rope to each
            for i, block in enumerate(self.backbone.blocks):
                if i >= self.num_frozen_blocks:
                    break
                x = block(x, rope)
            
            # Store seq_shape for use in forward_unfrozen_blocks
            self._last_seq_shape = seq_shape
            
            return x
    
    def extract_features_from_block(self, image_tensor, target_block_idx):
        """
        Extract features up to and including a specific block depth.
        
        This method is useful for analyzing which transformer blocks provide
        the best representations for semantic correspondence tasks. It extracts
        features cumulatively through blocks 0 to target_block_idx.
        
        DINOv3 requires RoPE (Rotary Position Embeddings) which are generated
        based on the spatial dimensions and passed to each transformer block.
        
        Args:
            image_tensor: Preprocessed image tensor (B, C, H, W)
            target_block_idx: Block index to extract through (inclusive)
                             - Positive: 0-indexed (e.g., 8 = blocks 0-8)
                             - Negative: from end (e.g., -1 = all blocks, -2 = all but last)
        
        Returns:
            torch.Tensor: Features after processing through specified blocks (B, N, C)
                         where N includes storage tokens (CLS + registers) and patch tokens
            
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
            # 1. Use DINOv3's prepare_tokens_with_masks for proper tokenization
            # This handles patch embedding, CLS token, and storage tokens
            x, seq_shape = self.backbone.prepare_tokens_with_masks(image_tensor)
            
            # 2. Generate RoPE embeddings for this spatial shape
            rope = self.backbone.rope_embed(H=seq_shape[0], W=seq_shape[1])
            
            # 3. Run through blocks 0 to target_block_idx (inclusive), passing rope to each
            for i in range(target_block_idx + 1):
                x = self.backbone.blocks[i](x, rope)
            
            # Store seq_shape for use in forward_from_block_features
            self._last_seq_shape = seq_shape
            
            return x
    
    def forward_from_block_features(self, block_features):
        """
        Apply norm and reshape block features for downstream tasks.
        
        This is useful when you want to use features from a specific block
        with the same post-processing as the standard pipeline.
        
        DINOv3 uses storage tokens (CLS + registers) which need to be removed
        to extract only the spatial patch tokens.
        
        Args:
            block_features: Features from extract_features_from_block (B, N, C)
                           where N includes storage tokens and patches
        
        Returns:
            torch.Tensor: Processed features (B, C, H, W) - ready for semantic correspondence
        """
        x = block_features.to(self.device)
        
        # Get seq_shape from cached value (set during extract_features_from_block)
        # or infer from standard_size
        if hasattr(self, '_last_seq_shape'):
            seq_shape = self._last_seq_shape
        else:
            # Fallback: compute from standard_size
            H = W = self.standard_size // self.patch_size
            seq_shape = (H, W)
            
            # Verify we have enough tokens
            n_patches = H * W
            B, N, C = x.shape
            if N < n_patches:
                raise ValueError(f"Not enough tokens! N={N}, Expected at least {n_patches} (H={H}, W={W})")
        
        # Apply final norm (no RoPE needed for norm)
        x = self.backbone.norm(x)
        
        # Remove storage tokens (CLS + Registers) to get patch tokens only
        # Patch tokens are always the LAST H*W tokens in the sequence
        H, W = seq_shape
        patch_tokens = x[:, -H*W:]  # (B, H*W, C)
        
        B, N, C = patch_tokens.shape
        
        # Reshape to spatial grid
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Apply dropout (only active during training)
        feature_map = self.dropout(feature_map)
        
        return feature_map
    
    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Run UNFROZEN blocks + norm with gradients.
        
        DINOv3 blocks require rope embeddings to be passed.
        """
        x = intermediate_features
        
        # Get seq_shape from cached value (set during extract_intermediate_features)
        # or infer from token count
        if hasattr(self, '_last_seq_shape'):
            seq_shape = self._last_seq_shape
        else:
            # Fallback: We verify against STANDARD_SIZE because we enforce it in extract_all_features
            # This is safer than inferring from N which behaves unpredictably with CLS/Register tokens
            H = W = self.standard_size // self.patch_size
            seq_shape = (H, W)
            
            # Verify we have enough tokens
            n_patches = H * W
            B, N, C = x.shape
            if N < n_patches:
                 raise ValueError(f"Not enough tokens! N={N}, Expected at least {n_patches} (H={H}, W={W})")
        
        # Generate RoPE embeddings (must be on same device as x)
        rope = self.backbone.rope_embed(H=seq_shape[0], W=seq_shape[1])
        
        # Run through UNFROZEN blocks with rope
        # Note: If LoRA is active, self.model IS the PeftModel which wraps self.backbone.
        # But here we are iterating over BLOCKS manually. 
        # CAUTION: If we iterate self.backbone.blocks, we are skipping the LoRA adapters 
        # if they are injected into the blocks! 
        # Peft injects modules into the model structure. So self.backbone.blocks IS modified 
        # and has LoraLayers in it. So efficient access via self.backbone is correct.
        
        for block in list(self.backbone.blocks)[-self.num_unfrozen_blocks:]:
            x = block(x, rope)
        
        # Apply final norm
        x = self.backbone.norm(x)
        
        # Remove storage tokens (CLS + Registers) to get patch tokens only
        # We assume patch tokens are always the LAST H*W tokens in ViT
        H, W = seq_shape
        patch_tokens = x[:, -H*W:]  # (B, H*W, C)
        
        B, N, C = patch_tokens.shape
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Apply dropout for regularization (only active during training)
        feature_map = self.dropout(feature_map)
        
        return feature_map
    
    def get_model_state(self):
        """Returns the dictionary containing model weights and metadata."""
        return {
            "backbone_state_dict": self.model.state_dict(),
            "embed_dim": self.model.embed_dim,
            "patch_size": self.patch_size,
            "model_name": self.model_name
        }
    
    def load_model_state(self, checkpoint):
        """Restores model weights from a checkpoint dictionary."""
        state_dict, format_info = extract_state_dict(checkpoint)
        print(f"Loading from {format_info}")
        self.model.load_state_dict(state_dict)
        print(f"Model state loaded for {self.model_name}")