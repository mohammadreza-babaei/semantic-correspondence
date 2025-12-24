import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import SamModel, SamConfig



# Standard image size for all feature extraction (divisible by patch_size=16)
# Reduced from 1024 for speed/memory efficiency if needed. 
# Default 1024 is native SAM resolution.
STANDARD_SIZE = 512 

class SAMAdapter:
    def __init__(
        self,
        model_name='facebook/sam-vit-base',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
    ):
        """
        Initializes the SAM model using HuggingFace Transformers.
        
        Args:
            model_name (str): HF Model ID. Options:
                              'facebook/sam-vit-base' (Recommended for speed/memory)
                              'facebook/sam-vit-large'
                              'facebook/sam-vit-huge'
            device (str): Computation device.
            num_unfrozen_blocks (int): Number of transformer blocks to unfreeze from the end.
        """
        self.device = device
        self.patch_size = 16  # SAM uses a patch size of 16 for its encoder
        self.standard_size = STANDARD_SIZE
        self.model_name = model_name
        
        print(f"Loading {model_name} on {self.device}...")
        
        # We load the full SAM model but will only use the vision_encoder part
        self.sam_model = SamModel.from_pretrained(model_name).to(self.device)
        self.sam_model.eval()
        
        self.embed_dim = self.sam_model.config.vision_config.hidden_size
        
        # Access the vision encoder
        self.model = self.sam_model.vision_encoder
        
        # Store number of unfrozen blocks for feature caching logic
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.layers) - num_unfrozen_blocks
        
        # Freeze all parameters first
        for param in self.sam_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer blocks in the vision encoder
        num_blocks = len(self.model.layers)
        print(f"Total transformer blocks: {num_blocks}")
        print(f"Frozen blocks: {self.num_frozen_blocks}, Unfrozen blocks: {num_unfrozen_blocks}")
        
        blocks_to_unfreeze = self.model.layers[-num_unfrozen_blocks:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.sam_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.sam_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        self.trainable_params = [p for p in self.sam_model.parameters() if p.requires_grad and p.is_leaf]
        
        # Override image size in config to avoid ValueError in patch_embed
        if self.standard_size != 1024:
            print(f"Resizing SAM vision encoder config to {self.standard_size}x{self.standard_size}")
            self.sam_model.vision_encoder.patch_embed.image_size = (self.standard_size, self.standard_size)
            if hasattr(self.sam_model.config, "vision_config"):
                self.sam_model.config.vision_config.image_size = self.standard_size

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
            
            # Keep spatial dimensions (B, H, W, C) - SAM layers expect this format
            # Note: SAM uses RELATIVE positional embeddings in the attention layers,
            # not absolute positional embeddings added here.
            # The positional encoding is handled internally by each transformer block.
            
            # Run through FROZEN blocks only
            for i, layer in enumerate(self.model.layers):
                if i >= self.num_frozen_blocks:
                    break
                x = layer(x)
            
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
        # Use slicing on ModuleList if possible, or iterate efficiently
        start_idx = self.num_frozen_blocks
        for i in range(start_idx, len(self.model.layers)):
            x = self.model.layers[i](x)
        
        # Apply neck if present
        if hasattr(self.model, 'neck') and hasattr(self.model.neck, '0'):
            # x is (B, H, W, C), need to reshape to (B, C, H, W) for neck
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # Apply neck (Conv2d layers)
            x = self.model.neck(x)
        else:
            # If no neck, just reshape from (B, H, W, C) to (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        
        # L2 normalize features
        feature_map = F.normalize(x, dim=1)
        
        return feature_map
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'backbone_state_dict': self.model.state_dict(),
            'sam_model_state_dict': self.sam_model.state_dict(),
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.sam_model.load_state_dict(checkpoint['sam_model_state_dict'])
        print(f"Checkpoint loaded: {path}")