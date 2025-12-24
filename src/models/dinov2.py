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


# Standard image size for all feature extraction (divisible by patch_size=14)
STANDARD_SIZE = 518 # Multiple of 14 (14*37=518), used in the DINOv2 paper

class DINOv2Adapter:
    def __init__(
        self,
        model_name='dinov2_vits14',
        weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
    ):
        """
        Initializes the DINOv2 model.
        
        Args:
            model_name (str): The DINOv2 model to load. Options: 
                              'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
            weights_path (str, optional): Path to custom .pth weights file. If None, loads pretrained from torch.hub.
            device (str): Computation device ('cuda' or 'cpu').
            num_unfrozen_blocks (int): Number of transformer blocks to unfreeze from the end.
        """
        self.standard_size = STANDARD_SIZE
        self.device = device
        self.patch_size = 14 # DINOv2 usually uses patch size 14
        self.model_name = model_name
        
        # Load model architecture and weights
        if weights_path is not None:
            # Load custom weights
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            print(f"Loading {model_name} architecture (no pretrained weights)...")
            self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False).to(self.device)
            
            print(f"Loading custom weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Clean state dictionary keys to handle various prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes that might be present in saved checkpoints
                k = k.replace("model.", "")
                k = k.replace("base_model.model.", "")
                k = k.replace("teacher.", "")
                k = k.replace("backbone.", "")
                new_state_dict[k] = v
            
            # Load the cleaned state dict
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Status: {msg}")
        else:
            # Load pretrained model from torch.hub
            print(f"Loading {model_name} with pretrained weights from torch.hub...")
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        
        self.model.eval() # Set to evaluation mode (frozen features)
        
        # Store number of unfrozen blocks for feature caching logic
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.blocks) - num_unfrozen_blocks
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer blocks
        num_blocks = len(self.model.blocks)
        print(f"Total transformer blocks: {num_blocks}")
        print(f"Frozen blocks: {self.num_frozen_blocks}, Unfrozen blocks: {num_unfrozen_blocks}")
        
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
            x = self.model.patch_embed(image_tensor)
            
            # Add CLS token
            cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Add position embeddings
            x = x + self.model.interpolate_pos_encoding(x, image_tensor.shape[2], image_tensor.shape[3])
            
            # Run through FROZEN blocks only
            for i, block in enumerate(self.model.blocks):
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
        for block in list(self.model.blocks)[-self.num_unfrozen_blocks:]:
            x = block(x)
        
        # Apply final norm
        x = self.model.norm(x)
        
        # Remove CLS token to get patch tokens only
        patch_tokens = x[:, 1:]  # (B, N, C)
        
        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # For 518x518 input with patch_size=14: 60x60
        assert H * W == N, f"Patch count {N} is not a perfect square"
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        return feature_map
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'backbone_state_dict': self.model.state_dict(),
            'embed_dim': self.model.embed_dim,
            'patch_size': self.patch_size
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['backbone_state_dict'])
        print(f"Checkpoint loaded: {path}")