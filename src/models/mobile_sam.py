import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from pathlib import Path
from .checkpoint_utils import extract_state_dict
from mobile_sam import sam_model_registry


# MobileSAM uses 1024 like SAM, but the encoder is TinyViT
STANDARD_SIZE = 1024

class MobileSAMAdapter:
    def __init__(
        self,
        model_name='vit_t', # MobileSAM is mapped to 'vit_t'
        weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2, # Number of STAGES to unfreeze (TinyViT has 4 stages)
    ):
        """
        Initializes the MobileSAM adapter.
        
        Args:
            model_name (str): Should be 'vit_t' for MobileSAM.
            weights_path (str): Path to 'mobile_sam.pt'. If None, initializes with random weights.
            device (str): 'cuda' or 'cpu'.
            num_unfrozen_blocks (int): Number of encoder stages to unfreeze (from the end).
        """
        self.device = device
        self.patch_size = 16
        self.standard_size = STANDARD_SIZE
        self.model_name = model_name # Internal identifier
        
        # 1. Robust Weight Loading
        # MobileSAM registry expects a specific key ('vit_t')
        registry_key = 'vit_t'
        
        print(f"Initializing MobileSAM ({self.model_name}) on {self.device}...")
        
        # Check if weights exist. If not, warn and set to None (Random Init)
        if weights_path is not None:
            if not os.path.exists(weights_path):
                print(f"Warning: MobileSAM weights not found at '{weights_path}'. Initializing with random weights.")
                weights_path = None
            else:
                print(f"Loading base weights from: {weights_path}")
        else:
            print("No base weights provided. Initializing with random weights (fine-tuning checkpoint expected later).")

        # Initialize Model (Full SAM)
        self.sam_model = sam_model_registry[registry_key](checkpoint=weights_path)
        self.sam_model.to(self.device)
        self.sam_model.eval() # Set to eval mode by default
        
        # We only care about the Image Encoder (TinyViT)
        self.model = self.sam_model.image_encoder

        # 2. Freeze / Unfreeze Logic
        # MobileSAM's TinyViT is structured into 'layers' (stages), not a flat list of blocks.
        # It typically has 4 stages.
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.model.layers) - num_unfrozen_blocks
        
        # First, freeze EVERYTHING
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
        # Unfreeze the last N stages of the encoder
        # Note: Unfreezing 2 stages of TinyViT is quite a lot (it's half the network).
        # You might want to experiment with num_unfrozen_blocks=1.
        print(f"Total Encoder Stages: {len(self.model.layers)}")
        print(f"Unfreezing last {num_unfrozen_blocks} stages...")
        
        stages_to_unfreeze = self.model.layers[-num_unfrozen_blocks:]
        for stage in stages_to_unfreeze:
            for param in stage.parameters():
                param.requires_grad = True
        
        # Unfreeze Neck (Conv layers after encoder) if present/needed? 
        # Usually for correspondence, we just want the backbone features.
        # But if you want to adapt the projection, unfreeze the neck (optional).
        # self.model.neck is usually simple convs.
        # for param in self.model.neck.parameters():
        #     param.requires_grad = True

        # Statistics
        trainable_params = sum(p.numel() for p in self.sam_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.sam_model.parameters())
        print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
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
        
        # We strictly resize to STANDARD_SIZE (512)
        # MobileSAM/TinyViT requires dimensions divisible by patch_size (16 or 32)
        # 512 is safe.
        target_h = self.standard_size
        target_w = self.standard_size
        
        resize_transform = T.Compose([
            T.Resize((target_h, target_w)), 
            T.ToTensor(),
            # SAM Mean/Std
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return resize_transform(img).unsqueeze(0).to(self.device)


    def extract_intermediate_features(self, image_tensor):
        """
        PASS-THROUGH: Returns the preprocessed image tensor itself.
        
        Since MobileSAM is lightweight, we don't cache partial layers. 
        We treat the raw image as the "intermediate" state to be saved by Trainer.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image (B, 3, H, W)
            
        Returns:
            torch.Tensor: Same as input.
        """
        # We simply return the image. The Trainer will save this to disk.
        # When loaded back in forward_unfrozen_blocks, it will be the input to the full encoder.
        return image_tensor

    
    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Runs the FULL MobileSAM Encoder.
        
        Args:
            intermediate_features (torch.Tensor): The preprocessed image (B, 3, H, W).
            
        Returns:
            torch.Tensor: Feature map (B, 256, H/16, W/16).
        """
        # Input is the image tensor loaded from cache
        x = intermediate_features.to(self.device)
        
        # Run the full TinyViT Image Encoder
        # Output of image_encoder is typically the feature map
        features = self.model(x)
        
        # MobileSAM/SAM output is usually (B, 256, 64, 64) for 1024 input
        # This is already in the correct format (B, C, H, W) for our loss function.
        return features

    
    def get_model_state(self):
        """
        Returns the dictionary containing model weights.
        We save the whole SAM state dict for compatibility.
        """
        return {
            "sam_model_state_dict": self.sam_model.state_dict(),
            "model_name": self.model_name
        }

    def load_model_state(self, checkpoint):
        """Restores model weights from a checkpoint dictionary."""
        state_dict, format_info = extract_state_dict(
            checkpoint,
            key_priority=['model_state', 'sam_model_state_dict']
        )
        print(f"Loading from {format_info}")
        self.sam_model.load_state_dict(state_dict)
        print(f"Model state loaded for {self.model_name}")