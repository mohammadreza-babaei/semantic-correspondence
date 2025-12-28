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
    ):
        """
        Initializes the DINOv3 model.
        
        Args:
            model_name (str): The DINOv3 model to load (e.g., 'dinov3_vits16').
            weights_path (str, optional): Path to local .pth file. If None, downloads from Hub.
            device (str): Computation device.
            num_unfrozen_blocks (int): Number of transformer blocks to unfreeze.
        """
        self.patch_size = 16
        self.standard_size = STANDARD_SIZE
        self.device = device
        self.model_name = model_name

        print(f"Loading local weights from: {weights_path}")
        self.model = torch.hub.load(REPO_SOURCE, model_name, pretrained=False).to(self.device)

        # --- FIX: CHECK IF PATH EXISTS BEFORE LOADING ---
        if weights_path is not None:
            if os.path.exists(weights_path):
                # 2. Load Weights (.pth)
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

                # 3. Clean Keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Remove Hugging Face specific prefixes
                    k = k.replace("model.", "") 
                    k = k.replace("base_model.model.", "")
                    # Remove standard DINO prefixes
                    k = k.replace("teacher.", "")
                    k = k.replace("backbone.", "") 
                    new_state_dict[k] = v
                
                # 4. Inject weights
                msg = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Weights loaded. Status: {msg}")
            else:
                print(f"Warning: Weights path '{weights_path}' not found. Initializing with random weights.")
        else:
            print("No weights_path provided. Initializing with random/base weights (expecting manual load).")
        # ------------------------------------------------
        
        # Feature caching setup
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.blocks) - num_unfrozen_blocks
        
        # 1. Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. Unfreeze the last N transformer blocks
        print(f"Total blocks: {len(self.model.blocks)}, Frozen: {self.num_frozen_blocks}, Unfrozen: {num_unfrozen_blocks}")
        
        blocks_to_unfreeze = list(self.model.blocks)[-num_unfrozen_blocks:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        # 3. Unfreeze final norm layer
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
            x, seq_shape = self.model.prepare_tokens_with_masks(image_tensor)
            
            # Step 2: Generate RoPE embeddings for this spatial shape
            rope = self.model.rope_embed(H=seq_shape[0], W=seq_shape[1])
            
            # Step 3: Run through FROZEN blocks only, passing rope to each
            for i, block in enumerate(self.model.blocks):
                if i >= self.num_frozen_blocks:
                    break
                x = block(x, rope)
            
            # Store seq_shape for use in forward_unfrozen_blocks
            self._last_seq_shape = seq_shape
            
            return x
    
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
        rope = self.model.rope_embed(H=seq_shape[0], W=seq_shape[1])
        
        # Run through UNFROZEN blocks with rope
        for block in list(self.model.blocks)[-self.num_unfrozen_blocks:]:
            x = block(x, rope)
        
        # Apply final norm
        x = self.model.norm(x)
        
        # Remove storage tokens (CLS + Registers) to get patch tokens only
        # We assume patch tokens are always the LAST H*W tokens in ViT
        H, W = seq_shape
        patch_tokens = x[:, -H*W:]  # (B, H*W, C)
        
        B, N, C = patch_tokens.shape
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return feature_map
    
    def get_model_state(self):
        """Returns the dictionary containing model weights and metadata."""
        return {
            "backbone_state_dict": self.model.state_dict(),
            "embed_dim": self.model.embed_dim,
            "patch_size": self.patch_size,
            "model_name": self.model_name
        }
    
    def load_model_state(self, state_dict):
        """Restores model weights from a state dictionary."""
        # Handle cases where the full checkpoint bundle is passed
        if "backbone_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["backbone_state_dict"])
        else:
            self.model.load_state_dict(state_dict)
        print(f"Model state loaded for {self.model_name}")